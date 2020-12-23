/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CloudViewer.h"
#include "CloudViewerCellPicker.h"
#include "util/Utils.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <QMenu>
#include <QAction>
#include <QtGui/QContextMenuEvent>
#include <QInputDialog>
#include <QtGui/QWheelEvent>
#include <QtGui/QKeyEvent>
#include <QColorDialog>
#include <QtGui/QVector3D>
#include <QMainWindow>
#include <QDebug>
#include <set>

#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkCubeSource.h>
#include <vtkGlyph3D.h>
#include <vtkGlyph3DMapper.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkImageData.h>
#include <vtkLookupTable.h>
#include <vtkTextureUnitManager.h>
#include <vtkJPEGReader.h>
#include <vtkBMPReader.h>
#include <vtkPNMReader.h>
#include <vtkPNGReader.h>
#include <vtkTIFFReader.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPointPicker.h>
#include <vtkTextActor.h>
#include <vtkOBBTree.h>
#include <vtkObjectFactory.h>
#include <vtkQuad.h>
#include <vtkTexture.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include "vtkImageMatSource.h"

// Standard VTK macro for *New ()
vtkStandardNewMacro (CloudViewerInteractorStyle);

CloudViewerInteractorStyle::CloudViewerInteractorStyle() :
    pcl::visualization::PCLVisualizerInteractorStyle(),
    viewer_(nullptr),
    NumberOfClicks(0),
    ResetPixelDistance(0),
    pointsHolder_(new pcl::PointCloud<pcl::PointXYZRGB>),
    orthoMode_(false)
{
    PreviousPosition[0] = PreviousPosition[1] = 0;
    PreviousMeasure[0] = PreviousMeasure[1] =  PreviousMeasure[2] = 0.0f;

    this->MotionFactor = 5;
}

void CloudViewerInteractorStyle::Rotate()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    int dx = rwi->GetEventPosition()[0] - rwi->GetLastEventPosition()[0];
    int dy = orthoMode_?0:rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1];

    int *size = this->CurrentRenderer->GetRenderWindow()->GetSize();

    double delta_elevation = -20.0 / size[1];
    double delta_azimuth = -20.0 / size[0];

    double rxf = dx * delta_azimuth * this->MotionFactor;
    double ryf = dy * delta_elevation * this->MotionFactor;

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    Q_ASSERT(camera);
    if(!orthoMode_)
    {
        camera->Azimuth(rxf);
        camera->Elevation(ryf);
        camera->OrthogonalizeViewUp();
    }
    else
    {
        camera->Roll(-rxf);
    }

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    if (rwi->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    //rwi->Render();
}

void CloudViewerInteractorStyle::setOrthoMode(bool enabled)
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkCamera *camera = CurrentRenderer->GetActiveCamera ();
    Q_ASSERT(camera);
    camera->SetParallelProjection (enabled);
    if(enabled)
    {
        double x,y,z;
        camera->GetFocalPoint(x, y, z);
        camera->SetPosition(x, y, z+(camera->GetDistance()<=5?5:camera->GetDistance()));
        camera->SetViewUp(1, 0, 0);
    }
    CurrentRenderer->SetActiveCamera (camera);
    orthoMode_ = enabled;
}

void CloudViewerInteractorStyle::OnMouseMove()
{
    if(this->CurrentRenderer &&
            this->CurrentRenderer->GetLayer() == 1 &&
            this->GetInteractor()->GetShiftKey() && this->GetInteractor()->GetControlKey() &&
            viewer_ &&
            viewer_->getLocators().size())
    {
        CloudViewerCellPicker * cellPicker = dynamic_cast<CloudViewerCellPicker*>(this->Interactor->GetPicker());
        if(cellPicker)
        {
            int pickPosition[2];
            this->GetInteractor()->GetEventPosition(pickPosition);
            this->Interactor->GetPicker()->Pick(pickPosition[0], pickPosition[1],
                    0,  // always zero.
                    this->CurrentRenderer);
            double picked[3];
            this->Interactor->GetPicker()->GetPickPosition(picked);

            //            UDEBUG("Control move! Picked value: %f %f %f", picked[0], picked[1], picked[2]);

            float textSize = 0.05;

            viewer_->removeCloud("interactor_points_alt");
            pointsHolder_->resize(2);
            pcl::PointXYZRGB pt;
            pt.x = picked[0];
            pt.y = picked[1];
            pt.z = picked[2];
            pointsHolder_->at(0) = pt;

            viewer_->removeLine("interactor_ray_alt");
            viewer_->removeText("interactor_ray_text_alt");

            // Intersect the locator with the line
            double length = 5.0;
            double pickedNormal[3];
            cellPicker->GetPickNormal(pickedNormal);
            double lineP0[3] = {picked[0], picked[1], picked[2]};
            double lineP1[3] = {picked[0]+pickedNormal[0]*length, picked[1]+pickedNormal[1]*length, picked[2]+pickedNormal[2]*length};
            vtkSmartPointer<vtkPoints> intersectPoints = vtkSmartPointer<vtkPoints>::New();

            viewer_->getLocators().begin().value()->IntersectWithLine(lineP0, lineP1, intersectPoints, nullptr);

            // Display list of intersections
            double intersection[3];
            double previous[3] = {picked[0], picked[1], picked[2]};
            for(int i = 0; i < intersectPoints->GetNumberOfPoints(); i++ )
            {
                intersectPoints->GetPoint(i, intersection);

                Eigen::Vector3f v(intersection[0]-previous[0], intersection[1]-previous[1], intersection[2]-previous[2]);
                float n = v.norm();
                if(n  > 0.01f)
                {
                    v/=n;
                    v *= n/2.0f;
                    pt.r = 125;
                    pt.g = 125;
                    pt.b = 125;
                    pt.x = intersection[0];
                    pt.y = intersection[1];
                    pt.z = intersection[2];
                    pointsHolder_->at(1) = pt;
                    viewer_->addOrUpdateText("interactor_ray_text_alt", QString("%1 m").arg(n),
                                             pointFrom(matrix4fFrom(previous[0]+v[0], previous[1]+v[1],previous[2]+v[2], 0, 0, 0)),
                            textSize,
                            Qt::gray);
                    viewer_->addOrUpdateLine("interactor_ray_alt",
                                             pointFrom(matrix4fFrom(previous[0], previous[1], previous[2], 0, 0, 0)),
                            pointFrom(matrix4fFrom(intersection[0], intersection[1], intersection[2], 0, 0, 0)),
                            Qt::gray);

                    previous[0] = intersection[0];
                    previous[1] = intersection[1];
                    previous[2] = intersection[2];
                    break;
                }
            }
            viewer_->addCloud("interactor_points_alt", pointsHolder_);
            viewer_->setCloudPointSize("interactor_points_alt", 15);
            viewer_->setCloudOpacity("interactor_points_alt", 0.5);
        }
    }
    // Forward events
    PCLVisualizerInteractorStyle::OnMouseMove();
}

void CloudViewerInteractorStyle::OnLeftButtonDown()
{
    // http://www.vtk.org/Wiki/VTK/Examples/Cxx/Interaction/DoubleClick
    // http://www.vtk.org/Wiki/VTK/Examples/Cxx/Interaction/PointPicker
    if(this->CurrentRenderer && this->CurrentRenderer->GetLayer() == 1)
    {
        this->NumberOfClicks++;
        int pickPosition[2];
        this->GetInteractor()->GetEventPosition(pickPosition);
        int xdist = pickPosition[0] - this->PreviousPosition[0];
        int ydist = pickPosition[1] - this->PreviousPosition[1];

        this->PreviousPosition[0] = pickPosition[0];
        this->PreviousPosition[1] = pickPosition[1];

        int moveDistance = (int)sqrt((double)(xdist*xdist + ydist*ydist));

        // Reset numClicks - If mouse moved further than resetPixelDistance
        if(moveDistance > this->ResetPixelDistance)
        {
            this->NumberOfClicks = 1;
        }

        if(this->NumberOfClicks >= 2)
        {
            this->NumberOfClicks = 0;
            this->Interactor->GetPicker()->Pick(pickPosition[0], pickPosition[1],
                    0,  // always zero.
                    this->CurrentRenderer);
            double picked[3];
            this->Interactor->GetPicker()->GetPickPosition(picked);
            //            UDEBUG("Double clicked! Picked value: %f %f %f", picked[0], picked[1], picked[2]);
            if(this->GetInteractor()->GetControlKey()==0)
            {
                vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
                Q_ASSERT(camera);
                double position[3];
                double focal[3];
                camera->GetPosition(position[0], position[1], position[2]);
                camera->GetFocalPoint(focal[0], focal[1], focal[2]);
                //camera->SetPosition (position[0] + (picked[0]-focal[0]), position[1] + (picked[1]-focal[1]), position[2] + (picked[2]-focal[2]));
                camera->SetFocalPoint (picked[0], picked[1], picked[2]);
                camera->OrthogonalizeViewUp();

                if (this->AutoAdjustCameraClippingRange)
                {
                    this->CurrentRenderer->ResetCameraClippingRange();
                }

                if (this->Interactor->GetLightFollowCamera())
                {
                    this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
                }
            }
            else if(viewer_)
            {
                viewer_->removeText("interactor_pose");
                viewer_->removeLine("interactor_line");
                viewer_->removeCloud("interactor_points");
                viewer_->removeLine("interactor_ray");
                viewer_->removeText("interactor_ray_text");
                viewer_->removeCloud("interactor_points_alt");
                viewer_->removeLine("interactor_ray_alt");
                viewer_->removeText("interactor_ray_text_alt");
                PreviousMeasure[0] = 0.0f;
                PreviousMeasure[1] = 0.0f;
                PreviousMeasure[2] = 0.0f;
            }
        }
        else if(this->GetInteractor()->GetControlKey() && viewer_)
        {
            this->Interactor->GetPicker()->Pick(pickPosition[0], pickPosition[1],
                    0,  // always zero.
                    this->CurrentRenderer);
            double picked[3];
            this->Interactor->GetPicker()->GetPickPosition(picked);

            //            UDEBUG("Shift clicked! Picked value: %f %f %f", picked[0], picked[1], picked[2]);

            float textSize = 0.05;

            viewer_->removeCloud("interactor_points");
            pointsHolder_->clear();
            pcl::PointXYZRGB pt;
            pt.x = picked[0];
            pt.y = picked[1];
            pt.z = picked[2];
            pointsHolder_->push_back(pt);

            viewer_->removeLine("interactor_ray");
            viewer_->removeText("interactor_ray_text");

            if(	PreviousMeasure[0] != 0.0f && PreviousMeasure[1] != 0.0f && PreviousMeasure[2] != 0.0f &&
                    viewer_->getAddedLines().find("interactor_line") == viewer_->getAddedLines().end())
            {
                viewer_->addOrUpdateLine("interactor_line",
                                         pointFrom(matrix4fFrom(PreviousMeasure[0], PreviousMeasure[1], PreviousMeasure[2], 0, 0, 0)),
                        pointFrom(matrix4fFrom(picked[0], picked[1], picked[2], 0, 0, 0)),
                        Qt::red);
                pt.x = PreviousMeasure[0];
                pt.y = PreviousMeasure[1];
                pt.z = PreviousMeasure[2];
                pointsHolder_->push_back(pt);

                Eigen::Vector3f v(picked[0]-PreviousMeasure[0], picked[1]-PreviousMeasure[1], picked[2]-PreviousMeasure[2]);
                float n = v.norm();
                v/=n;
                v *= n/2.0f;
                viewer_->addOrUpdateText("interactor_pose", QString("%1 m").arg(n),
                                         pointFrom(matrix4fFrom(PreviousMeasure[0]+v[0], PreviousMeasure[1]+v[1],PreviousMeasure[2]+v[2], 0, 0, 0)),
                        textSize,
                        Qt::red);
            }
            else
            {
                viewer_->removeText("interactor_pose");
                viewer_->removeLine("interactor_line");
            }
            PreviousMeasure[0] = picked[0];
            PreviousMeasure[1] = picked[1];
            PreviousMeasure[2] = picked[2];

            viewer_->addCloud("interactor_points", pointsHolder_);
            viewer_->setCloudPointSize("interactor_points", 15);
            viewer_->setCloudOpacity("interactor_points", 0.5);
        }
    }

    // Forward events
    PCLVisualizerInteractorStyle::OnLeftButtonDown();
}

CloudViewer::CloudViewer(QWidget *parent, CloudViewerInteractorStyle * style) :
    QVTKOpenGLWidget(parent),
    _aLockCamera(nullptr),
    _aFollowCamera(nullptr),
    _aResetCamera(nullptr),
    _aLockViewZ(nullptr),
    _aCameraOrtho(nullptr),
    _aShowTrajectory(nullptr),
    _aSetTrajectorySize(nullptr),
    _aClearTrajectory(nullptr),
    _aShowFrustum(nullptr),
    _aSetFrustumScale(nullptr),
    _aSetFrustumColor(nullptr),
    _aShowGrid(nullptr),
    _aSetGridCellCount(nullptr),
    _aSetGridCellSize(nullptr),
    _aShowNormals(nullptr),
    _aSetNormalsStep(nullptr),
    _aSetNormalsScale(nullptr),
    _aSetBackgroundColor(nullptr),
    _aSetRenderingRate(nullptr),
    _aSetLighting(nullptr),
    _aSetFlatShading(nullptr),
    _aSetEdgeVisibility(nullptr),
    _aBackfaceCulling(nullptr),
    _menu(nullptr),
    _trajectory(new pcl::PointCloud<pcl::PointXYZ>),
    _maxTrajectorySize(100),
    _frustumScale(0.5f),
    _frustumColor(Qt::gray),
    _gridCellCount(50),
    _gridCellSize(1),
    _normalsStep(1),
    _normalsScale(0.2f),
    _buildLocator(false),
    _lastCameraOrientation(0,0,0),
    _lastCameraPose(0,0,0),
    _defaultBgColor(Qt::black),
    _currentBgColor(Qt::black),
    _frontfaceCulling(false),
    _renderingRate(5.0),
    _octomapActor(nullptr)
{
    //	UDEBUG("");
    this->setMinimumSize(200, 200);

    int argc = 0;
    Q_ASSERT(style!=0);
    style->setCloudViewer(this);
    vtkNew<vtkGenericOpenGLRenderWindow> window;
    vtkNew<vtkRenderer> r;
    window->AddRenderer(r.Get());

    _visualizer = new pcl::visualization::PCLVisualizer(
                argc,
                0,
                r.Get(),
                window.Get(),
                "PCLVisualizer",
                style,
                false);

    _visualizer->setShowFPS(false);
    int viewport;
    _visualizer->createViewPort (0,0,1.0, 1.0, viewport); // all 3d objects here
    _visualizer->createViewPort (0,0,1.0, 1.0, viewport); // text overlay
    _visualizer->getRendererCollection()->InitTraversal ();
    vtkRenderer* renderer = nullptr;
    int i =0;
    while ((renderer = _visualizer->getRendererCollection()->GetNextItem ()) != nullptr)
    {
        renderer->SetLayer(i);
        if(i==1)
        {
            _visualizer->getInteractorStyle()->SetDefaultRenderer(renderer);
        }
        ++i;
    }
    //    if (renderer == nullptr)
    //    {
    //        _visualizer->getInteractorStyle()->SetDefaultRenderer(r.Get());
    //    }
    _visualizer->getRenderWindow()->SetNumberOfLayers(3);

    this->SetRenderWindow(_visualizer->getRenderWindow());

    // Replaced by the second line, to avoid a crash in Mac OS X on close, as well as
    // the "Invalid drawable" warning when the view is not visible.
    //_visualizer->setupInteractor(this->GetInteractor(), this->GetRenderWindow());
    this->GetInteractor()->SetInteractorStyle (_visualizer->getInteractorStyle());
    // setup a simple point picker
    vtkSmartPointer<vtkPointPicker> pp = vtkSmartPointer<vtkPointPicker>::New ();
    //	UDEBUG("pick tolerance=%f", pp->GetTolerance());
    pp->SetTolerance (pp->GetTolerance()/2.0);
    this->GetInteractor()->SetPicker (pp);

    setRenderingRate(_renderingRate);

    _visualizer->setCameraPosition(
                -1, 0, 0,
                0, 0, 0,
                0, 0, 1, 1);
//#ifndef _WIN32
    // Crash on startup on Windows (vtk issue)
    this->addOrUpdateCoordinate("reference", Eigen::Matrix4f::Identity(), 0.2);
//#endif

    //setup menu/actions
    createMenu();

    setMouseTracking(false);
}

CloudViewer::~CloudViewer()
{
    //    UDEBUG("");
    this->clear();
    delete _visualizer;
    //	UDEBUG("");
}

void CloudViewer::clear()
{
    this->removeAllClouds();
    this->removeAllGraphs();
    this->removeAllCoordinates();
    this->removeAllLines();
    this->removeAllFrustums();
    this->removeAllTexts();
    this->removeOccupancyGridMap();

    this->addOrUpdateCoordinate("reference", Eigen::Matrix4f::Identity(), 0.2);
    _lastPose.setZero();
    if(_aLockCamera->isChecked() || _aFollowCamera->isChecked())
    {
        resetCamera();
    }
    this->clearTrajectory();
}

void CloudViewer::createMenu()
{
    _aLockCamera = new QAction("Lock target", this);
    _aLockCamera->setCheckable(true);
    _aLockCamera->setChecked(false);
    _aFollowCamera = new QAction("Follow", this);
    _aFollowCamera->setCheckable(true);
    _aFollowCamera->setChecked(true);
    QAction * freeCamera = new QAction("Free", this);
    freeCamera->setCheckable(true);
    freeCamera->setChecked(false);
    _aLockViewZ = new QAction("Lock view Z", this);
    _aLockViewZ->setCheckable(true);
    _aLockViewZ->setChecked(false);
    _aCameraOrtho = new QAction("Ortho mode", this);
    _aCameraOrtho->setCheckable(true);
    _aCameraOrtho->setChecked(false);
    _aResetCamera = new QAction("Reset position", this);
    _aShowTrajectory= new QAction("Show trajectory", this);
    _aShowTrajectory->setCheckable(true);
    _aShowTrajectory->setChecked(true);
    _aSetTrajectorySize = new QAction("Set trajectory size...", this);
    _aClearTrajectory = new QAction("Clear trajectory", this);
    _aShowFrustum= new QAction("Show frustum", this);
    _aShowFrustum->setCheckable(true);
    _aShowFrustum->setChecked(false);
    _aSetFrustumScale = new QAction("Set frustum scale...", this);
    _aSetFrustumColor = new QAction("Set frustum color...", this);
    _aShowGrid = new QAction("Show grid", this);
    _aShowGrid->setCheckable(true);
    _aSetGridCellCount = new QAction("Set cell count...", this);
    _aSetGridCellSize = new QAction("Set cell size...", this);
    _aShowNormals = new QAction("Show normals", this);
    _aShowNormals->setCheckable(true);
    _aSetNormalsStep = new QAction("Set normals step...", this);
    _aSetNormalsScale = new QAction("Set normals scale...", this);
    _aSetBackgroundColor = new QAction("Set background color...", this);
    _aSetRenderingRate = new QAction("Set rendering rate...", this);
    _aSetLighting = new QAction("Lighting", this);
    _aSetLighting->setCheckable(true);
    _aSetLighting->setChecked(false);
    _aSetFlatShading = new QAction("Flat Shading", this);
    _aSetFlatShading->setCheckable(true);
    _aSetFlatShading->setChecked(false);
    _aSetEdgeVisibility = new QAction("Show edges", this);
    _aSetEdgeVisibility->setCheckable(true);
    _aSetEdgeVisibility->setChecked(false);
    _aBackfaceCulling = new QAction("Backface culling", this);
    _aBackfaceCulling->setCheckable(true);
    _aBackfaceCulling->setChecked(true);
    _aPolygonPicking = new QAction("Polygon picking", this);
    _aPolygonPicking->setCheckable(true);
    _aPolygonPicking->setChecked(false);

    QMenu * cameraMenu = new QMenu("Camera", this);
    cameraMenu->addAction(_aLockCamera);
    cameraMenu->addAction(_aFollowCamera);
    cameraMenu->addAction(freeCamera);
    cameraMenu->addSeparator();
    cameraMenu->addAction(_aLockViewZ);
    cameraMenu->addAction(_aCameraOrtho);
    cameraMenu->addAction(_aResetCamera);
    QActionGroup * group = new QActionGroup(this);
    group->addAction(_aLockCamera);
    group->addAction(_aFollowCamera);
    group->addAction(freeCamera);

    QMenu * trajectoryMenu = new QMenu("Trajectory", this);
    trajectoryMenu->addAction(_aShowTrajectory);
    trajectoryMenu->addAction(_aSetTrajectorySize);
    trajectoryMenu->addAction(_aClearTrajectory);

    QMenu * frustumMenu = new QMenu("Frustum", this);
    frustumMenu->addAction(_aShowFrustum);
    frustumMenu->addAction(_aSetFrustumScale);
    frustumMenu->addAction(_aSetFrustumColor);

    QMenu * gridMenu = new QMenu("Grid", this);
    gridMenu->addAction(_aShowGrid);
    gridMenu->addAction(_aSetGridCellCount);
    gridMenu->addAction(_aSetGridCellSize);

    QMenu * normalsMenu = new QMenu("Normals", this);
    normalsMenu->addAction(_aShowNormals);
    normalsMenu->addAction(_aSetNormalsStep);
    normalsMenu->addAction(_aSetNormalsScale);

    //menus
    _menu = new QMenu(this);
    _menu->addMenu(cameraMenu);
    _menu->addMenu(trajectoryMenu);
    _menu->addMenu(frustumMenu);
    _menu->addMenu(gridMenu);
    _menu->addMenu(normalsMenu);
    _menu->addAction(_aSetBackgroundColor);
    _menu->addAction(_aSetRenderingRate);
    _menu->addAction(_aSetLighting);
    _menu->addAction(_aSetFlatShading);
    _menu->addAction(_aSetEdgeVisibility);
    _menu->addAction(_aBackfaceCulling);
    _menu->addAction(_aPolygonPicking);
}

void CloudViewer::saveSettings(QSettings & settings, const QString & group) const
{
    if(!group.isEmpty())
    {
        settings.beginGroup(group);
    }

    float poseX, poseY, poseZ, focalX, focalY, focalZ, upX, upY, upZ;
    this->getCameraPosition(poseX, poseY, poseZ, focalX, focalY, focalZ, upX, upY, upZ);
    QVector3D pose(poseX, poseY, poseZ);
    QVector3D focal(focalX, focalY, focalZ);
    if(!this->isCameraFree())
    {
        // make camera position relative to target
        Eigen::Matrix4f T = this->getTargetPose();
        Eigen::Vector3f pt = pointFrom(T);
        if(this->isCameraTargetLocked())
        {
            T = matrix4fFrom(pt.x(), pt.y(), pt.z(), 0,0,0);
        }
        Eigen::Matrix4f F = matrix4fFrom(focalX, focalY, focalZ, 0,0,0);
        Eigen::Matrix4f P = matrix4fFrom(poseX, poseY, poseZ, 0,0,0);
        Eigen::Matrix4f newFocal = T.inverse() * F;
        Eigen::Vector3f newFocalPt = pointFrom(newFocal);
        Eigen::Matrix4f newPose = newFocal * F.inverse() * P;
        Eigen::Vector3f newPosePt = pointFrom(newPose);
        pose = QVector3D(newPosePt.x(), newPosePt.y(), newPosePt.z());
        focal = QVector3D(newFocalPt.x(), newFocalPt.y(), newFocalPt.z());
    }
    settings.setValue("camera_pose", pose);
    settings.setValue("camera_focal", focal);
    settings.setValue("camera_up", QVector3D(upX, upY, upZ));

    settings.setValue("grid", this->isGridShown());
    settings.setValue("grid_cell_count", this->getGridCellCount());
    settings.setValue("grid_cell_size", (double)this->getGridCellSize());

    settings.setValue("normals", this->isNormalsShown());
    settings.setValue("normals_step", this->getNormalsStep());
    settings.setValue("normals_scale", (double)this->getNormalsScale());

    settings.setValue("trajectory_shown", this->isTrajectoryShown());
    settings.setValue("trajectory_size", this->getTrajectorySize());

    settings.setValue("frustum_shown", this->isFrustumShown());
    settings.setValue("frustum_scale", this->getFrustumScale());
    settings.setValue("frustum_color", this->getFrustumColor());

    settings.setValue("camera_target_locked", this->isCameraTargetLocked());
    settings.setValue("camera_target_follow", this->isCameraTargetFollow());
    settings.setValue("camera_free", this->isCameraFree());
    settings.setValue("camera_lockZ", this->isCameraLockZ());
    settings.setValue("camera_ortho", this->isCameraOrtho());

    settings.setValue("bg_color", this->getDefaultBackgroundColor());
    settings.setValue("rendering_rate", this->getRenderingRate());
    if(!group.isEmpty())
    {
        settings.endGroup();
    }
}

void CloudViewer::loadSettings(QSettings & settings, const QString & group)
{
    if(!group.isEmpty())
    {
        settings.beginGroup(group);
    }

    float poseX, poseY, poseZ, focalX, focalY, focalZ, upX, upY, upZ;
    this->getCameraPosition(poseX, poseY, poseZ, focalX, focalY, focalZ, upX, upY, upZ);
    QVector3D pose(poseX, poseY, poseZ), focal(focalX, focalY, focalZ), up(upX, upY, upZ);
    pose = settings.value("camera_pose", pose).value<QVector3D>();
    focal = settings.value("camera_focal", focal).value<QVector3D>();
    up = settings.value("camera_up", up).value<QVector3D>();
    this->setCameraPosition(pose.x(),pose.y(),pose.z(), focal.x(),focal.y(),focal.z(), up.x(),up.y(),up.z());

    this->setGridShown(settings.value("grid", this->isGridShown()).toBool());
    this->setGridCellCount(settings.value("grid_cell_count", this->getGridCellCount()).toUInt());
    this->setGridCellSize(settings.value("grid_cell_size", this->getGridCellSize()).toFloat());

    this->setNormalsShown(settings.value("normals", this->isNormalsShown()).toBool());
    this->setNormalsStep(settings.value("normals_step", this->getNormalsStep()).toInt());
    this->setNormalsScale(settings.value("normals_scale", this->getNormalsScale()).toFloat());

    this->setTrajectoryShown(settings.value("trajectory_shown", this->isTrajectoryShown()).toBool());
    this->setTrajectorySize(settings.value("trajectory_size", this->getTrajectorySize()).toUInt());

    this->setFrustumShown(settings.value("frustum_shown", this->isFrustumShown()).toBool());
    this->setFrustumScale(settings.value("frustum_scale", this->getFrustumScale()).toDouble());
    this->setFrustumColor(settings.value("frustum_color", this->getFrustumColor()).value<QColor>());

    this->setCameraTargetLocked(settings.value("camera_target_locked", this->isCameraTargetLocked()).toBool());
    this->setCameraTargetFollow(settings.value("camera_target_follow", this->isCameraTargetFollow()).toBool());
    if(settings.value("camera_free", this->isCameraFree()).toBool())
    {
        this->setCameraFree();
    }
    this->setCameraLockZ(settings.value("camera_lockZ", this->isCameraLockZ()).toBool());
    this->setCameraOrtho(settings.value("camera_ortho", this->isCameraOrtho()).toBool());

    this->setDefaultBackgroundColor(settings.value("bg_color", this->getDefaultBackgroundColor()).value<QColor>());

    this->setRenderingRate(settings.value("rendering_rate", this->getRenderingRate()).toDouble());

    if(!group.isEmpty())
    {
        settings.endGroup();
    }

    this->update();
}

bool CloudViewer::updateCloudPose(const QString & id,
                                  const Eigen::Matrix4f &pose)
{
    if(_addedClouds.contains(id))
    {
        //UDEBUG("Updating pose %s to %s", id.c_str(), pose.prettyPrint().c_str());
        bool samePose = _addedClouds.find(id).value() == pose;
        Eigen::Affine3f posef = affine3fFrom(pose);
        if(samePose ||
                _visualizer->updatePointCloudPose(id.toStdString(), posef))
        {
            _addedClouds.find(id).value() = pose;
            if(!samePose)
            {
                QString idNormals = id+"-normals";
                if(_addedClouds.contains(idNormals))
                {
                    _visualizer->updatePointCloudPose(idNormals.toStdString(), posef);
                    _addedClouds.find(idNormals).value() = pose;
                }
            }
            return true;
        }
    }
    return false;
}

class PointCloudColorHandlerIntensityField : public pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>
{
    typedef pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::PointCloud PointCloud;
    typedef PointCloud::Ptr PointCloudPtr;
    typedef PointCloud::ConstPtr PointCloudConstPtr;

public:
    typedef boost::shared_ptr<PointCloudColorHandlerIntensityField > Ptr;
    typedef boost::shared_ptr<const PointCloudColorHandlerIntensityField > ConstPtr;

    /** \brief Constructor. */
    PointCloudColorHandlerIntensityField (const PointCloudConstPtr &cloud) :
        pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::PointCloudColorHandler (cloud)
    {
        field_idx_  = pcl::getFieldIndex (*cloud, "intensity");
        if (field_idx_ != -1)
            capable_ = true;
        else
            capable_ = false;
    }

    /** \brief Empty destructor */
    virtual ~PointCloudColorHandlerIntensityField () {}

    /** \brief Obtain the actual color for the input dataset as vtk scalars.
     * \param[out] scalars the output scalars containing the color for the dataset
     * \return true if the operation was successful (the handler is capable and
     * the input cloud was given as a valid pointer), false otherwise
     */
    virtual bool
    getColor (vtkSmartPointer<vtkDataArray> &scalars) const
    {
        if (!capable_ || !cloud_)
            return (false);

        if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        scalars->SetNumberOfComponents (3);

        vtkIdType nr_points = cloud_->width * cloud_->height;
        // Allocate enough memory to hold all colors
        float * intensities = new float[nr_points];
        float intensity;
        size_t point_offset = cloud_->fields[field_idx_].offset;
        size_t j = 0;

        // If XYZ present, check if the points are invalid
        int x_idx = pcl::getFieldIndex (*cloud_, "x");
        if (x_idx != -1)
        {
            float x_data, y_data, z_data;
            size_t x_point_offset = cloud_->fields[x_idx].offset;

            // Color every point
            for (vtkIdType cp = 0; cp < nr_points; ++cp,
                 point_offset += cloud_->point_step,
                 x_point_offset += cloud_->point_step)
            {
                // Copy the value at the specified field
                memcpy (&intensity, &cloud_->data[point_offset], sizeof (float));

                memcpy (&x_data, &cloud_->data[x_point_offset], sizeof (float));
                memcpy (&y_data, &cloud_->data[x_point_offset + sizeof (float)], sizeof (float));
                memcpy (&z_data, &cloud_->data[x_point_offset + 2 * sizeof (float)], sizeof (float));

                if (!std::isfinite (x_data) || !std::isfinite (y_data) || !std::isfinite (z_data))
                    continue;

                intensities[j++] = intensity;
            }
        }
        // No XYZ data checks
        else
        {
            // Color every point
            for (vtkIdType cp = 0; cp < nr_points; ++cp, point_offset += cloud_->point_step)
            {
                // Copy the value at the specified field
                memcpy (&intensity, &cloud_->data[point_offset], sizeof (float));

                intensities[j++] = intensity;
            }
        }
        if (j != 0)
        {
            // Allocate enough memory to hold all colors
            unsigned char* colors = new unsigned char[j * 3];
            float min, max;
            findMinMax(intensities, j, min, max);
            for(size_t k=0; k<j; ++k)
            {
                colors[k*3+0] = colors[k*3+1] = colors[k*3+2] = max>0?(unsigned char)(intensities[k]/max*255.0f):0;
            }
            reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (j);
            reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetArray (colors, j*3, 0, vtkUnsignedCharArray::VTK_DATA_ARRAY_DELETE);
        }
        else
            reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (0);
        //delete [] colors;
        delete [] intensities;
        return (true);
    }

protected:
    /** \brief Get the name of the class. */
    virtual std::string
    getName () const { return ("PointCloudColorHandlerIntensityField"); }

    /** \brief Get the name of the field used. */
    virtual std::string
    getFieldName () const { return ("intensity"); }
};

bool CloudViewer::addCloud(const QString &id,
                           const pcl::PCLPointCloud2Ptr & binaryCloud,
                           const Eigen::Matrix4f & pose,
                           bool rgb,
                           bool hasNormals,
                           bool hasIntensity,
                           const QColor & color)
{
    int previousColorIndex = -1;
    if(_addedClouds.contains(id))
    {
        previousColorIndex = _visualizer->getColorHandlerIndex(id.toStdString());
        this->removeCloud(id);
    }

    Eigen::Vector4f origin = vector4fZeroFrom(pose);
    Eigen::Quaternionf orientation = Eigen::Quaternionf(affine3fFrom(pose).linear());

    if(hasNormals && _aShowNormals->isChecked())
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointNormal>);
        pcl::fromPCLPointCloud2 (*binaryCloud, *cloud_xyz);
        QString idNormals = id + "-normals";
        if(_visualizer->addPointCloudNormals<pcl::PointNormal>(cloud_xyz, _normalsStep, _normalsScale, idNormals.toStdString(), 0))
        {
            _visualizer->updatePointCloudPose(idNormals.toStdString(), affine3fFrom(pose));
            _addedClouds.insert(idNormals, pose);
        }
    }

    // add random color channel
    pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::Ptr colorHandler;
    colorHandler.reset (new pcl::visualization::PointCloudColorHandlerRandom<pcl::PCLPointCloud2> (binaryCloud));
    if(_visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1))
    {
        QColor c = Qt::gray;
        if(color.isValid())
        {
            c = color;
        }
        colorHandler.reset (new pcl::visualization::PointCloudColorHandlerCustom<pcl::PCLPointCloud2> (binaryCloud, c.red(), c.green(), c.blue()));
        _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);

        // x,y,z
        colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "x"));
        _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
        colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "y"));
        _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
        colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "z"));
        _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);

        if(rgb)
        {
            //rgb
            colorHandler.reset(new pcl::visualization::PointCloudColorHandlerRGBField<pcl::PCLPointCloud2>(binaryCloud));
            _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
        }
        else if(hasIntensity)
        {
            //intensity
            colorHandler.reset(new PointCloudColorHandlerIntensityField(binaryCloud));
            _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
        }
        else if(previousColorIndex == 5)
        {
            previousColorIndex = -1;
        }

        if(hasNormals)
        {
            //normals
            colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "normal_x"));
            _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
            colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "normal_y"));
            _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
            colorHandler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PCLPointCloud2> (binaryCloud, "normal_z"));
            _visualizer->addPointCloud (binaryCloud, colorHandler, origin, orientation, id.toStdString(), 1);
        }
        else if(previousColorIndex > 5)
        {
            previousColorIndex = -1;
        }

        if(previousColorIndex>=0)
        {
            _visualizer->updateColorHandlerIndex(id.toStdString(), previousColorIndex);
        }
        else if(rgb)
        {
            _visualizer->updateColorHandlerIndex(id.toStdString(), 5);
        }
        else if(hasNormals)
        {
            _visualizer->updateColorHandlerIndex(id.toStdString(), hasIntensity?8:7);
        }
        else if(hasIntensity)
        {
            _visualizer->updateColorHandlerIndex(id.toStdString(), 5);
        }
        else if(color.isValid())
        {
            _visualizer->updateColorHandlerIndex(id.toStdString(), 1);
        }

        _addedClouds.insert(id, pose);
        return true;
    }
    return false;
}

bool CloudViewer::addCloud(
        const QString & id,
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & cloud,
        const Eigen::Matrix4f & pose,
        const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, true, true, false, color);
}

bool CloudViewer::addCloud(
        const QString & id,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
        const Eigen::Matrix4f & pose,
        const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, true, false, false, color);
}

bool CloudViewer::addCloud(const QString &id,
                           const pcl::PointCloud<pcl::PointXYZINormal>::Ptr & cloud,
                           const Eigen::Matrix4f & pose,
                           const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, false, true, true, color);
}

bool CloudViewer::addCloud(const QString &id,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud,
                           const Eigen::Matrix4f & pose,
                           const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, false, false, true, color);
}

bool CloudViewer::addCloud(const QString &id,
                           const pcl::PointCloud<pcl::PointNormal>::Ptr & cloud,
                           const Eigen::Matrix4f & pose,
                           const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, false, true, false, color);
}

bool CloudViewer::addCloud(const QString &id,
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
                           const Eigen::Matrix4f & pose,
                           const QColor & color)
{
    pcl::PCLPointCloud2Ptr binaryCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *binaryCloud);
    return addCloud(id, binaryCloud, pose, false, false, false, color);
}

bool CloudViewer::addCloudMesh(const QString & id,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
                               const QVector<pcl::Vertices> &polygons,
                               const Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        this->removeCloud(id);
    }

    //	UDEBUG("Adding %s with %d points and %d polygons", id.c_str(), (int)cloud->size(), (int)polygons.size());
    if(_visualizer->addPolygonMesh<pcl::PointXYZ>(cloud, polygons.toStdVector(), id.toStdString(), 1))
    {
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetAmbient(0.5);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
        _visualizer->updatePointCloudPose(id.toStdString(), affine3fFrom(pose));
        if(_buildLocator)
        {
            vtkSmartPointer<vtkOBBTree> tree = vtkSmartPointer<vtkOBBTree>::New();
            tree->SetDataSet(_visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetMapper()->GetInput());
            tree->BuildLocator();
            _locators.insert(id, tree);
        }
        _addedClouds.insert(id, pose);
        return true;
    }
    return false;
}

bool CloudViewer::addCloudMesh(const QString & id,
                               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                               const QVector<pcl::Vertices> &polygons,
                               const Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        this->removeCloud(id);
    }

    //	UDEBUG("Adding %s with %d points and %d polygons", id.c_str(), (int)cloud->size(), (int)polygons.size());
    if(_visualizer->addPolygonMesh<pcl::PointXYZRGB>(cloud, polygons.toStdVector(), id.toStdString(), 1))
    {
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetAmbient(0.5);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
        _visualizer->updatePointCloudPose(id.toStdString(), affine3fFrom(pose));
        if(_buildLocator)
        {
            vtkSmartPointer<vtkOBBTree> tree = vtkSmartPointer<vtkOBBTree>::New();
            tree->SetDataSet(_visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetMapper()->GetInput());
            tree->BuildLocator();
            _locators.insert(id, tree);
        }
        _addedClouds.insert(id, pose);
        return true;
    }
    return false;
}

bool CloudViewer::addCloudMesh(
        const QString & id,
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & cloud,
        const QVector<pcl::Vertices> & polygons,
        const Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        this->removeCloud(id);
    }

    //	UDEBUG("Adding %s with %d points and %d polygons", id.c_str(), (int)cloud->size(), (int)polygons.size());
    if(_visualizer->addPolygonMesh<pcl::PointXYZRGBNormal>(cloud, polygons.toStdVector(), id.toStdString(), 1))
    {
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetAmbient(0.5);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
        _visualizer->updatePointCloudPose(id.toStdString(), affine3fFrom(pose));
        if(_buildLocator)
        {
            vtkSmartPointer<vtkOBBTree> tree = vtkSmartPointer<vtkOBBTree>::New();
            tree->SetDataSet(_visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetMapper()->GetInput());
            tree->BuildLocator();
            _locators.insert(id, tree);
        }
        _addedClouds.insert(id, pose);
        return true;
    }
    return false;
}

bool CloudViewer::addCloudMesh(
        const QString & id,
        const pcl::PolygonMesh::Ptr & mesh,
        const Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        this->removeCloud(id);
    }

    //	UDEBUG("Adding %s with %d polygons", id.c_str(), (int)mesh->polygons.size());
    if(_visualizer->addPolygonMesh(*mesh, id.toStdString(), 1))
    {
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetAmbient(0.5);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
        _visualizer->updatePointCloudPose(id.toStdString(), affine3fFrom(pose));
        if(_buildLocator)
        {
            vtkSmartPointer<vtkOBBTree> tree = vtkSmartPointer<vtkOBBTree>::New();
            tree->SetDataSet(_visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetMapper()->GetInput());
            tree->BuildLocator();
            _locators.insert(id, tree);
        }
        _addedClouds.insert(id, pose);
        return true;
    }

    return false;
}

bool CloudViewer::addCloudTextureMesh(
        const QString & id,
        const pcl::TextureMesh::Ptr & textureMesh,
        const cv::Mat & texture,
        const Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        this->removeCloud(id);
    }

    //	UDEBUG("Adding %s", id.c_str());
    if(this->addTextureMesh(*textureMesh, texture, id, 1))
    {
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG);
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
        if(!textureMesh->cloud.is_dense)
        {
            _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetTexture()->SetInterpolate(1);
            _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetTexture()->SetBlendingMode(vtkTexture::VTK_TEXTURE_BLENDING_MODE_REPLACE);
        }
        _visualizer->updatePointCloudPose(id.toStdString(), affine3fFrom(pose));
        if(_buildLocator)
        {
            vtkSmartPointer<vtkOBBTree> tree = vtkSmartPointer<vtkOBBTree>::New();
            tree->SetDataSet(_visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetMapper()->GetInput());
            tree->BuildLocator();
            _locators.insert(id, tree);
        }
        _addedClouds.insert(id, pose);
        return true;
    }
    return false;
}

bool CloudViewer::addTextureMesh (
        const pcl::TextureMesh &mesh,
        const cv::Mat & image,
        const QString &id,
        int viewport)
{
    // Copied from PCL 1.8, modified to ignore vertex color and accept only one material (loaded from memory instead of file)

    pcl::visualization::CloudActorMap::iterator am_it = _visualizer->getCloudActorMap()->find(id.toStdString());
    if (am_it != _visualizer->getCloudActorMap()->end ())
    {
        PCL_ERROR ("[PCLVisualizer::addTextureMesh] A shape with id <%s> already exists!"
                   " Please choose a different id and retry.\n",
                   id.toStdString().c_str ());
        return (false);
    }
    // no texture materials --> exit
    if (mesh.tex_materials.size () == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures found!\n");
        return (false);
    }
    else if (mesh.tex_materials.size() > 1)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] only one material per mesh is supported!\n");
        return (false);
    }
    // polygons are mapped to texture materials
    if (mesh.tex_materials.size () != mesh.tex_polygons.size ())
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Materials number %lu differs from polygons number %lu!\n",
                  mesh.tex_materials.size (), mesh.tex_polygons.size ());
        return (false);
    }
    // each texture material should have its coordinates set
    if (mesh.tex_materials.size () != mesh.tex_coordinates.size ())
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Coordinates number %lu differs from materials number %lu!\n",
                  mesh.tex_coordinates.size (), mesh.tex_materials.size ());
        return (false);
    }
    // total number of vertices
    std::size_t nb_vertices = 0;
    for (std::size_t i = 0; i < mesh.tex_polygons.size (); ++i)
        nb_vertices+= mesh.tex_polygons[i].size ();
    // no vertices --> exit
    if (nb_vertices == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No vertices found!\n");
        return (false);
    }
    // total number of coordinates
    std::size_t nb_coordinates = 0;
    for (std::size_t i = 0; i < mesh.tex_coordinates.size (); ++i)
        nb_coordinates+= mesh.tex_coordinates[i].size ();
    // no texture coordinates --> exit
    if (nb_coordinates == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures coordinates found!\n");
        return (false);
    }

    // Create points from mesh.cloud
    vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New ();
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    bool has_color = false;
    vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New ();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2 (mesh.cloud, *cloud);
    // no points --> exit
    if (cloud->points.size () == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Cloud is empty!\n");
        return (false);
    }
    pcl::visualization::PCLVisualizer::convertToVtkMatrix (cloud->sensor_origin_, cloud->sensor_orientation_, transformation);
    poly_points->SetNumberOfPoints (cloud->points.size ());
    for (std::size_t i = 0; i < cloud->points.size (); ++i)
    {
        const pcl::PointXYZ &p = cloud->points[i];
        poly_points->InsertPoint (i, p.x, p.y, p.z);
    }

    //create polys from polyMesh.tex_polygons
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New ();
    for (std::size_t i = 0; i < mesh.tex_polygons.size (); i++)
    {
        for (std::size_t j = 0; j < mesh.tex_polygons[i].size (); j++)
        {
            std::size_t n_points = mesh.tex_polygons[i][j].vertices.size ();
            polys->InsertNextCell (int (n_points));
            for (std::size_t k = 0; k < n_points; k++)
                polys->InsertCellPoint (mesh.tex_polygons[i][j].vertices[k]);
        }
    }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPolys (polys);
    polydata->SetPoints (poly_points);
    if (has_color)
        polydata->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
#if VTK_MAJOR_VERSION < 6
    mapper->SetInput (polydata);
#else
    mapper->SetInputData (polydata);
#endif

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New ();
    vtkTextureUnitManager* tex_manager = vtkOpenGLRenderWindow::SafeDownCast (_visualizer->getRenderWindow())->GetTextureUnitManager ();
    if (!tex_manager)
        return (false);

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New ();
    // fill vtkTexture from pcl::TexMaterial structure
    vtkSmartPointer<vtkImageMatSource> cvImageToVtk = vtkSmartPointer<vtkImageMatSource>::New();
    cvImageToVtk->SetImage(image);
    cvImageToVtk->Update();
    texture->SetInputConnection(cvImageToVtk->GetOutputPort());

    // set texture coordinates
    vtkSmartPointer<vtkFloatArray> coordinates = vtkSmartPointer<vtkFloatArray>::New ();
    coordinates->SetNumberOfComponents (2);
    coordinates->SetNumberOfTuples (mesh.tex_coordinates[0].size ());
    for (std::size_t tc = 0; tc < mesh.tex_coordinates[0].size (); ++tc)
    {
        const Eigen::Vector2f &uv = mesh.tex_coordinates[0][tc];
        coordinates->SetTuple2 (tc, (double)uv[0], (double)uv[1]);
    }
    coordinates->SetName ("TCoords");
    polydata->GetPointData ()->SetTCoords(coordinates);
    // apply texture
    actor->SetTexture (texture);

    // set mapper
    actor->SetMapper (mapper);

    //_visualizer->addActorToRenderer (actor, viewport);
    // Add it to all renderers
    _visualizer->getRendererCollection()->InitTraversal ();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = _visualizer->getRendererCollection()->GetNextItem ()) != nullptr)
    {
        // Should we add the actor to all renderers?
        if (viewport == 0)
        {
            renderer->AddActor (actor);
        }
        else if (viewport == i)               // add the actor only to the specified viewport
        {
            renderer->AddActor (actor);
        }
        ++i;
    }

    // Save the pointer/ID pair to the global actor map
    (*_visualizer->getCloudActorMap())[id.toStdString()].actor = actor;

    // Save the viewpoint transformation matrix to the global actor map
    (*_visualizer->getCloudActorMap())[id.toStdString()].viewpoint_transformation_ = transformation;

    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetAmbient(0.5);
    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG);
    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
    _visualizer->getCloudActorMap()->find(id.toStdString())->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
    return true;
}

bool CloudViewer::addOccupancyGridMap(
        const cv::Mat & map8U,
        float resolution, // cell size
        float xMin,
        float yMin,
        float opacity)
{
    Q_ASSERT(map8U.channels() == 1 && map8U.type() == CV_8U);

    float xSize = float(map8U.cols) * resolution;
    float ySize = float(map8U.rows) * resolution;

    //	UDEBUG("resolution=%f, xSize=%f, ySize=%f, xMin=%f, yMin=%f", resolution, xSize, ySize, xMin, yMin);
    if(_visualizer->getCloudActorMap()->find("map") != _visualizer->getCloudActorMap()->end())
    {
        _visualizer->removePointCloud("map");
    }

    if(xSize > 0.0f && ySize > 0.0f)
    {
        pcl::TextureMeshPtr mesh(new pcl::TextureMesh());
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.push_back(pcl::PointXYZ(xMin,       yMin,       0));
        cloud.push_back(pcl::PointXYZ(xSize+xMin, yMin,       0));
        cloud.push_back(pcl::PointXYZ(xSize+xMin, ySize+yMin, 0));
        cloud.push_back(pcl::PointXYZ(xMin,       ySize+yMin, 0));
        pcl::toPCLPointCloud2(cloud, mesh->cloud);

        std::vector<pcl::Vertices> polygons(1);
        polygons[0].vertices.push_back(0);
        polygons[0].vertices.push_back(1);
        polygons[0].vertices.push_back(2);
        polygons[0].vertices.push_back(3);
        polygons[0].vertices.push_back(0);
        mesh->tex_polygons.push_back(polygons);

        // default texture materials parameters
        pcl::TexMaterial material;
        material.tex_file = "";
        mesh->tex_materials.push_back(material);

#if PCL_VERSION_COMPARE(>=, 1, 8, 0)
        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > coordinates;
#else
        std::vector<Eigen::Vector2f> coordinates;
#endif
        coordinates.push_back(Eigen::Vector2f(0,1));
        coordinates.push_back(Eigen::Vector2f(1,1));
        coordinates.push_back(Eigen::Vector2f(1,0));
        coordinates.push_back(Eigen::Vector2f(0,0));
        mesh->tex_coordinates.push_back(coordinates);

        this->addTextureMesh(*mesh, map8U, "map", 1);
        setCloudOpacity("map", opacity);
    }
    return true;
}

void CloudViewer::removeOccupancyGridMap()
{
    if(_visualizer->getCloudActorMap()->find("map") != _visualizer->getCloudActorMap()->end())
    {
        _visualizer->removePointCloud("map");
    }
}

void CloudViewer::addOrUpdateCoordinate(
        const QString & id,
        const Eigen::Matrix4f & transform,
        double scale,
        bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeCoordinate(id);

    if(!transform.isZero())
    {
        _coordinates.insert(id);
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
        _visualizer->addCoordinateSystem(scale, affine3fFrom(transform), id.toStdString(), foreground?2:1);
#else
        // Well, on older versions, just update the main coordinate
        _visualizer->addCoordinateSystem(scale, transform.toEigen3f(), 0);
#endif
    }
}

bool CloudViewer::updateCoordinatePose(
        const QString & id,
        const Eigen::Matrix4f & pose)
{
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
    if(_coordinates.find(id) != _coordinates.end() && !pose.isZero())
    {
        //		UDEBUG("Updating pose %s to %s", id.c_str(), pose.prettyPrint().c_str());
        return _visualizer->updateCoordinateSystemPose(id.toStdString(), affine3fFrom(pose));
    }
#else
    //	UERROR("CloudViewer::updateCoordinatePose() is not available on PCL < 1.7.2");
#endif
    return false;
}

void CloudViewer::removeCoordinate(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_coordinates.find(id) != _coordinates.end())
    {
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
        _visualizer->removeCoordinateSystem(id.toStdString());
#else
        // Well, on older versions, just update the main coordinate
        _visualizer->removeCoordinateSystem(0);
#endif
        _coordinates.remove(id);
    }
}

void CloudViewer::removeAllCoordinates(const QString & prefix)
{
    QSet<QString> coordinates = _coordinates;
    for(QSet<QString>::iterator iter = coordinates.begin(); iter!=coordinates.end(); ++iter)
    {
        if(prefix.isEmpty() || iter->contains(prefix))
        {
            this->removeCoordinate(*iter);
        }
    }
    Q_ASSERT(!prefix.isEmpty() || _coordinates.empty());
}

void CloudViewer::addOrUpdateLine(
        const QString & id,
        const Eigen::Vector3f & from,
        const Eigen::Vector3f & to,
        const QColor & color,
        bool arrow,
        bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeLine(id);

    if(!from.isZero() && !to.isZero())
    {
        _lines.insert(id);

        QColor c = Qt::gray;
        if(color.isValid())
        {
            c = color;
        }

        pcl::PointXYZ pt1(from.x(), from.y(), from.z());
        pcl::PointXYZ pt2(to.x(), to.y(), to.z());

        if(arrow)
        {
            _visualizer->addArrow(pt2, pt1, c.redF(), c.greenF(), c.blueF(), false, id.toStdString(), foreground?2:1);
        }
        else
        {
            _visualizer->addLine(pt2, pt1, c.redF(), c.greenF(), c.blueF(), id.toStdString(), foreground?2:1);
        }
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, c.alphaF(), id.toStdString());
    }
}

void CloudViewer::removeLine(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_lines.find(id) != _lines.end())
    {
        _visualizer->removeShape(id.toStdString());
        _lines.remove(id);
    }
}

void CloudViewer::removeAllLines()
{
    QSet<QString> arrows = _lines;
    for (QSet<QString>::iterator i = arrows.begin(); i != arrows.end(); i++)
    {
        removeLine(*i);
    }
    Q_ASSERT(_lines.empty());
}

void CloudViewer::addOrUpdateSphere(
        const QString & id,
        const Eigen::Vector3f & pose,
        float radius,
        const QColor & color,
        bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeSphere(id);

    if(!pose.isZero())
    {
        _spheres.insert(id);

        QColor c = Qt::gray;
        if(color.isValid())
        {
            c = color;
        }

        pcl::PointXYZ center(pose.x(), pose.y(), pose.z());
        _visualizer->addSphere(center, radius, c.redF(), c.greenF(), c.blueF(), id.toStdString(), foreground?2:1);
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, c.alphaF(), id.toStdString());
    }
}

void CloudViewer::removeSphere(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_spheres.contains(id))
    {
        _visualizer->removeShape(id.toStdString());
        _spheres.remove(id);
    }
}

void CloudViewer::removeAllSpheres()
{
    QSet<QString> spheres = _spheres;
    for (QSet<QString>::iterator i = spheres.begin(); i != spheres.end(); i++)
    {
        removeSphere(*i);
    }
    Q_ASSERT(_spheres.empty());
}

void CloudViewer::addOrUpdateCube(
        const QString & id,
        const Eigen::Matrix4f & pose,
        float width,
        float height,
        float depth,
        const QColor & color,
        bool wireframe,
        bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeCube(id);

    if(!pose.isZero())
    {
        _cubes.insert(id);

        QColor c = Qt::gray;
        if(color.isValid())
        {
            c = color;
        }
        _visualizer->addCube(pointFrom(pose), quaternionfFrom(pose), width, height, depth, id.toStdString(), foreground?2:1);
        if(wireframe)
        {
            _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id.toStdString());
        }
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, c.redF(), c.greenF(), c.blueF(), id.toStdString());
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, c.alphaF(), id.toStdString());
    }
}

void CloudViewer::removeCube(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_cubes.contains(id))
    {
        _visualizer->removeShape(id.toStdString());
        _cubes.remove(id);
    }
}

void CloudViewer::removeAllCubes()
{
    QSet<QString> cubes = _cubes;
    for (QSet<QString>::iterator i = cubes.begin(); i != cubes.end(); i++)
    {
        removeCube(*i);
    }
    Q_ASSERT(_cubes.empty());
}

void CloudViewer::addOrUpdateQuad(
        const QString & id,
        const Eigen::Matrix4f & pose,
        float width,
        float height,
        const QColor & color,
        bool foreground)
{
    addOrUpdateQuad(id, pose, width/2.0f, width/2.0f, height/2.0f, height/2.0f, color, foreground);
}

void CloudViewer::addOrUpdateQuad(
        const QString & id,
        const Eigen::Matrix4f & pose,
        float widthLeft,
        float widthRight,
        float heightBottom,
        float heightTop,
        const QColor & color,
        bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeQuad(id);

    if(!pose.isZero())
    {
        _quads.insert(id);

        QColor c = Qt::gray;
        if(color.isValid())
        {
            c = color;
        }

        // Create four points (must be in counter clockwise order)
        double p0[3] = {0.0, -widthLeft, heightTop};
        double p1[3] = {0.0, -widthLeft, -heightBottom};
        double p2[3] = {0.0, widthRight, -heightBottom};
        double p3[3] = {0.0, widthRight, heightTop};

        // Add the points to a vtkPoints object
        vtkSmartPointer<vtkPoints> points =
                vtkSmartPointer<vtkPoints>::New();
        points->InsertNextPoint(p0);
        points->InsertNextPoint(p1);
        points->InsertNextPoint(p2);
        points->InsertNextPoint(p3);

        // Create a quad on the four points
        vtkSmartPointer<vtkQuad> quad =
                vtkSmartPointer<vtkQuad>::New();
        quad->GetPointIds()->SetId(0,0);
        quad->GetPointIds()->SetId(1,1);
        quad->GetPointIds()->SetId(2,2);
        quad->GetPointIds()->SetId(3,3);

        // Create a cell array to store the quad in
        vtkSmartPointer<vtkCellArray> quads =
                vtkSmartPointer<vtkCellArray>::New();
        quads->InsertNextCell(quad);

        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> polydata =
                vtkSmartPointer<vtkPolyData>::New();

        // Add the points and quads to the dataset
        polydata->SetPoints(points);
        polydata->SetPolys(quads);

        // Setup actor and mapper
        vtkSmartPointer<vtkPolyDataMapper> mapper =
                vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
        mapper->SetInput(polydata);
#else
        mapper->SetInputData(polydata);
#endif

        vtkSmartPointer<vtkLODActor> actor =
                vtkSmartPointer<vtkLODActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(c.redF(), c.greenF(), c.blueF());

        //_visualizer->addActorToRenderer (actor, viewport);
        // Add it to all renderers
        _visualizer->getRendererCollection()->InitTraversal ();
        vtkRenderer* renderer = nullptr;
        int i = 0;
        while ((renderer = _visualizer->getRendererCollection()->GetNextItem ()) != nullptr)
        {
            if ((foreground?2:1) == i)               // add the actor only to the specified viewport
            {
                renderer->AddActor (actor);
            }
            ++i;
        }

        // Save the pointer/ID pair to the global actor map
        (*_visualizer->getCloudActorMap())[id.toStdString()].actor = actor;

        // Save the viewpoint transformation matrix to the global actor map
        vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New ();
        pcl::visualization::PCLVisualizer::convertToVtkMatrix (pose, transformation);
        (*_visualizer->getCloudActorMap())[id.toStdString()].viewpoint_transformation_ = transformation;
        (*_visualizer->getCloudActorMap())[id.toStdString()].actor->SetUserMatrix (transformation);
        (*_visualizer->getCloudActorMap())[id.toStdString()].actor->Modified ();

        (*_visualizer->getCloudActorMap())[id.toStdString()].actor->GetProperty()->SetLighting(false);
        _visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, c.alphaF(), id.toStdString());
    }
}

void CloudViewer::removeQuad(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_quads.contains(id))
    {
        _visualizer->removeShape(id.toStdString());
        _quads.remove(id);
    }
}

void CloudViewer::removeAllQuads()
{
    QSet<QString> quads = _quads;
    for (QSet<QString>::iterator i = quads.begin(); i != quads.end(); i++)
    {
        removeQuad(*i);
    }
    Q_ASSERT(_quads.empty());
}

static const float frustum_vertices[] = {
    0.0f,  0.0f, 0.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f};

static const int frustum_indices[] = {
    1, 2, 3, 4, 1, 0, 2, 0, 3, 0, 4};

void CloudViewer::addOrUpdateFrustum(
        const QString & id,
        const Eigen::Matrix4f & transform,
        const Eigen::Matrix4f & localTransform,
        double scale,
        const QColor & color)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

#if PCL_VERSION_COMPARE(<, 1, 7, 2)
    this->removeFrustum(id);
#endif

    if(!transform.isZero())
    {
        if(_frustums.contains(id))
        {
            _frustums.insert(id, Eigen::Matrix4f());

            int frustumSize = sizeof(frustum_vertices)/sizeof(float);
            Q_ASSERT(frustumSize>0 && frustumSize % 3 == 0);
            frustumSize/=3;
            pcl::PointCloud<pcl::PointXYZ> frustumPoints;
            frustumPoints.resize(frustumSize);
            float scaleX = 0.5f * scale;
            float scaleY = 0.4f * scale; //4x3 arbitrary ratio
            float scaleZ = 0.3f * scale;
            QColor c = Qt::gray;
            if(color.isValid())
            {
                c = color;
            }
            Eigen::Matrix4f opticalRotInv;
            opticalRotInv << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;

#if PCL_VERSION_COMPARE(<, 1, 7, 2)
            Eigen::Affine3f t = (transform*localTransform*opticalRotInv).toEigen3f();
#else
            Eigen::Affine3f t = affine3fFrom(localTransform*opticalRotInv);
#endif
            for(int i=0; i<frustumSize; ++i)
            {
                frustumPoints[i].x = frustum_vertices[i*3]*scaleX;
                frustumPoints[i].y = frustum_vertices[i*3+1]*scaleY;
                frustumPoints[i].z = frustum_vertices[i*3+2]*scaleZ;
                frustumPoints[i] = pcl::transformPoint(frustumPoints[i], t);
            }

            pcl::PolygonMesh mesh;
            pcl::Vertices vertices;
            vertices.vertices.resize(sizeof(frustum_indices)/sizeof(int));
            for(unsigned int i=0; i<vertices.vertices.size(); ++i)
            {
                vertices.vertices[i] = frustum_indices[i];
            }
            pcl::toPCLPointCloud2(frustumPoints, mesh.cloud);
            mesh.polygons.push_back(vertices);
            _visualizer->addPolylineFromPolygonMesh(mesh, id.toStdString(), 1);
            _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, c.redF(), c.greenF(), c.blueF(), id.toStdString());
            _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, c.alphaF(), id.toStdString());
        }
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
        if(!this->updateFrustumPose(id, transform))
        {
            //			UERROR("Failed updating pose of frustum %s!?", id.c_str());
        }
#endif
    }
    else
    {
        removeFrustum(id);
    }
}

bool CloudViewer::updateFrustumPose(
        const QString & id,
        const Eigen::Matrix4f & pose)
{
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
    QMap<QString, Eigen::Matrix4f>::iterator iter=_frustums.find(id);
    if(iter != _frustums.end() && !pose.isZero())
    {
        if(iter.value() == pose)
        {
            // same pose, just return
            return true;
        }

        pcl::visualization::ShapeActorMap::iterator am_it = _visualizer->getShapeActorMap()->find (id.toStdString());

        vtkActor* actor;

        if (am_it == _visualizer->getShapeActorMap()->end ())
            return (false);
        else
            actor = vtkActor::SafeDownCast (am_it->second);

        if (!actor)
            return (false);

        vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New ();

        pcl::visualization::PCLVisualizer::convertToVtkMatrix (pose, matrix);

        actor->SetUserMatrix (matrix);
        actor->Modified ();

        iter.value() = pose;

        return true;
    }
#else
    //	UERROR("updateFrustumPose() cannot be used with PCL<1.7.2. Use addOrUpdateFrustum() instead.");
#endif
    return false;
}

void CloudViewer::removeFrustum(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_frustums.contains(id))
    {
        _visualizer->removeShape(id.toStdString());
        _frustums.remove(id);
    }
}

void CloudViewer::removeAllFrustums(bool exceptCameraReference)
{
    QMap<QString, Eigen::Matrix4f> frustums = _frustums;
    for(QMap<QString, Eigen::Matrix4f>::iterator iter = frustums.begin(); iter!=frustums.end(); ++iter)
    {
        if(!exceptCameraReference || iter.key().contains("reference_frustum"))
        {
            this->removeFrustum(iter.key());
        }
    }
    Q_ASSERT(exceptCameraReference || _frustums.empty());
}

void CloudViewer::addOrUpdateGraph(
        const QString & id,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr & graph,
        const QColor & color)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeGraph(id);

    if(graph->size())
    {
        _graphes.insert(id);

        pcl::PolygonMesh mesh;
        pcl::Vertices vertices;
        vertices.vertices.resize(graph->size());
        for(unsigned int i=0; i<vertices.vertices.size(); ++i)
        {
            vertices.vertices[i] = i;
        }
        pcl::toPCLPointCloud2(*graph, mesh.cloud);
        mesh.polygons.push_back(vertices);
        _visualizer->addPolylineFromPolygonMesh(mesh, id.toStdString(), 1);
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.redF(), color.greenF(), color.blueF(), id.toStdString());
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, color.alphaF(), id.toStdString());

        this->addCloud(id+"_nodes", graph, Eigen::Matrix4f::Identity(), color);
        this->setCloudPointSize(id+"_nodes", 5);
    }
}

void CloudViewer::removeGraph(const QString & id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_graphes.contains(id))
    {
        _visualizer->removeShape(id.toStdString());
        _graphes.remove(id);
        removeCloud(id+"_nodes");
    }
}

void CloudViewer::removeAllGraphs()
{
    QSet<QString> graphes = _graphes;
    for (QSet<QString>::iterator i = graphes.begin(); i != graphes.end(); i++)
    {
        removeGraph(*i);
    }
    Q_ASSERT(_graphes.empty());
}

void CloudViewer::addOrUpdateText(const QString &id,
                                  const QString &text,
                                  const Eigen::Vector3f & position,
                                  double scale,
                                  const QColor & color,
                                  bool foreground)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    removeText(id);

    if(!position.isZero())
    {
        _texts.insert(id);
        _visualizer->addText3D(
                    text.toStdString(),
                    pcl::PointXYZ(position.x(), position.y(), position.z()),
                    scale,
                    color.redF(),
                    color.greenF(),
                    color.blueF(),
                    id.toStdString(),
                    foreground?2:1);
    }
}

void CloudViewer::removeText(const QString &id)
{
    if(id.isEmpty())
    {
        //		UERROR("id should not be empty!");
        return;
    }

    if(_texts.contains(id))
    {
        _visualizer->removeText3D(id.toStdString());
        _texts.remove(id);
    }
}

void CloudViewer::removeAllTexts()
{
    QSet<QString> texts = _texts;
    for (QSet<QString>::iterator i = texts.begin(); i != texts.end(); i++)
    {
        removeText(*i);
    }
    Q_ASSERT(_texts.empty());
}

bool CloudViewer::isTrajectoryShown() const
{
    return _aShowTrajectory->isChecked();
}

unsigned int CloudViewer::getTrajectorySize() const
{
    return _maxTrajectorySize;
}

void CloudViewer::setTrajectoryShown(bool shown)
{
    _aShowTrajectory->setChecked(shown);
}

void CloudViewer::setTrajectorySize(unsigned int value)
{
    _maxTrajectorySize = value;
}

void CloudViewer::clearTrajectory()
{
    _trajectory->clear();
    _visualizer->removeShape("trajectory");
    this->update();
}

bool CloudViewer::isFrustumShown() const
{
    return _aShowFrustum->isChecked();
}

float CloudViewer::getFrustumScale() const
{
    return _frustumScale;
}

QColor CloudViewer::getFrustumColor() const
{
    return _frustumColor;
}

void CloudViewer::setFrustumShown(bool shown)
{
    if(!shown)
    {
        QMap<QString, Eigen::Matrix4f> frustumsCopy = _frustums;
        for(QMap<QString, Eigen::Matrix4f>::iterator iter=frustumsCopy.begin(); iter!=frustumsCopy.end(); ++iter)
        {
            if(iter.key().contains("reference_frustum"))
            {
                this->removeFrustum(iter.key());
            }
        }
        QSet<QString> linesCopy = _lines;
        for(QSet<QString>::iterator iter=linesCopy.begin(); iter!=linesCopy.end(); ++iter)
        {
            if(iter->contains("reference_frustum_line"))
            {
                this->removeLine(*iter);
            }
        }
        this->update();
    }
    _aShowFrustum->setChecked(shown);
}

void CloudViewer::setFrustumScale(float value)
{
    _frustumScale = value;
}

void CloudViewer::setFrustumColor(QColor value)
{
    if(!value.isValid())
    {
        value = Qt::gray;
    }
    for(QMap<QString, Eigen::Matrix4f>::iterator iter=_frustums.begin(); iter!=_frustums.end(); ++iter)
    {
        _visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, value.redF(), value.greenF(), value.blueF(), iter.key().toStdString());
    }
    this->update();
    _frustumColor = value;
}

void CloudViewer::resetCamera()
{
    _lastCameraOrientation= _lastCameraPose = cv::Vec3f(0,0,0);
    if((_aFollowCamera->isChecked() || _aLockCamera->isChecked()) && !_lastPose.isZero())
    {
        // reset relative to last current pose
        Eigen::Matrix4f translation = translationFrom(rotationFrom(_lastPose) * matrix4fFrom(-1, 0, 0));
        Eigen::Vector3f pt = transformPoint(pointFrom(_lastPose), translation);
        if(_aCameraOrtho->isChecked())
        {
            _visualizer->setCameraPosition(
                        pt.x(), pt.y(), pt.z()+5,
                        pt.x(), pt.y(), pt.z(),
                        1, 0, 0, 1);
        }
        else if(_aLockViewZ->isChecked())
        {
            _visualizer->setCameraPosition(
                        pt.x(), pt.y(), pt.z(),
                        pt.x(), pt.y(), pt.z(),
                        0, 0, 1, 1);
        }
        else
        {
            _visualizer->setCameraPosition(
                        pt.x(), pt.y(), pt.z(),
                        pt.x(), pt.y(), pt.z(),
                        _lastPose(2, 0), _lastPose(2, 1), _lastPose(2, 2), 1);
        }
    }
    else if(_aCameraOrtho->isChecked())
    {
        _visualizer->setCameraPosition(
                    0, 0, 5,
                    0, 0, 0,
                    1, 0, 0, 1);
    }
    else
    {
        _visualizer->setCameraPosition(
                    -1, 0, 0,
                    0, 0, 0,
                    0, 0, 1, 1);
    }
    this->update();
}

void CloudViewer::removeAllClouds()
{
    _addedClouds.clear();
    _locators.clear();
    _visualizer->removeAllPointClouds();
}


bool CloudViewer::removeCloud(const QString & id)
{
    bool success = _visualizer->removePointCloud(id.toStdString());
    _visualizer->removePointCloud((id+"-normals").toStdString());
    _addedClouds.remove(id); // remove after visualizer
    _addedClouds.remove(id+"-normals");
    _locators.remove(id);
    return success;
}

bool CloudViewer::getPose(const QString & id, Eigen::Matrix4f & pose)
{
    if(_addedClouds.contains(id))
    {
        pose = _addedClouds.value(id);
        return true;
    }
    return false;
}

Eigen::Matrix4f CloudViewer::getTargetPose() const
{
    if(_lastPose.isZero())
    {
        return Eigen::Matrix4f::Identity();
    }
    return _lastPose;
}

QString CloudViewer::getIdByActor(vtkProp * actor) const
{
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    for(pcl::visualization::CloudActorMap::iterator iter=cloudActorMap->begin(); iter!=cloudActorMap->end(); ++iter)
    {
        if(iter->second.actor.GetPointer() == actor)
        {
            return QString::fromStdString(iter->first);
        }
    }

#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
    // getShapeActorMap() not available in version < 1.7.2
    pcl::visualization::ShapeActorMapPtr shapeActorMap = _visualizer->getShapeActorMap();
    for(pcl::visualization::ShapeActorMap::iterator iter=shapeActorMap->begin(); iter!=shapeActorMap->end(); ++iter)
    {
        if(iter->second.GetPointer() == actor)
        {
            std::string id = iter->first;
            while(id.size() && id.at(id.size()-1) == '*')
            {
                id.erase(id.size()-1);
            }

            return QString::fromStdString(id);
        }
    }
#endif
    return "";
}

QColor CloudViewer::getColor(const QString & id)
{
    QColor color;
    pcl::visualization::CloudActorMap::iterator iter = _visualizer->getCloudActorMap()->find(id.toStdString());
    if(iter != _visualizer->getCloudActorMap()->end())
    {
        double r,g,b,a;
        iter->second.actor->GetProperty()->GetColor(r,g,b);
        a = iter->second.actor->GetProperty()->GetOpacity();
        color.setRgbF(r, g, b, a);
    }
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
    // getShapeActorMap() not available in version < 1.7.2
    else
    {
        QString idLayer1 = id+"*";
        QString idLayer2 = id+"**";
        pcl::visualization::ShapeActorMap::iterator iter = _visualizer->getShapeActorMap()->find(id.toStdString());
        if(iter == _visualizer->getShapeActorMap()->end())
        {
            iter = _visualizer->getShapeActorMap()->find(idLayer1.toStdString());
            if(iter == _visualizer->getShapeActorMap()->end())
            {
                iter = _visualizer->getShapeActorMap()->find(idLayer2.toStdString());
            }
        }
        if(iter != _visualizer->getShapeActorMap()->end())
        {
            vtkActor * actor = vtkActor::SafeDownCast(iter->second);
            if(actor)
            {
                double r,g,b,a;
                actor->GetProperty()->GetColor(r,g,b);
                a = actor->GetProperty()->GetOpacity();
                color.setRgbF(r, g, b, a);
            }
        }
    }
#endif
    return color;
}

void CloudViewer::setColor(const QString & id, const QColor & color)
{
    pcl::visualization::CloudActorMap::iterator iter = _visualizer->getCloudActorMap()->find(id.toStdString());
    if(iter != _visualizer->getCloudActorMap()->end())
    {
        iter->second.actor->GetProperty()->SetColor(color.redF(),color.greenF(),color.blueF());
        iter->second.actor->GetProperty()->SetOpacity(color.alphaF());
    }
#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
    // getShapeActorMap() not available in version < 1.7.2
    else
    {
        QString idLayer1 = id+"*";
        QString idLayer2 = id+"**";
        pcl::visualization::ShapeActorMap::iterator iter = _visualizer->getShapeActorMap()->find(id.toStdString());
        if(iter == _visualizer->getShapeActorMap()->end())
        {
            iter = _visualizer->getShapeActorMap()->find(idLayer1.toStdString());
            if(iter == _visualizer->getShapeActorMap()->end())
            {
                iter = _visualizer->getShapeActorMap()->find(idLayer2.toStdString());
            }
        }
        if(iter != _visualizer->getShapeActorMap()->end())
        {
            vtkActor * actor = vtkActor::SafeDownCast(iter->second);
            if(actor)
            {
                actor->GetProperty()->SetColor(color.redF(),color.greenF(),color.blueF());
                actor->GetProperty()->SetOpacity(color.alphaF());
            }
        }
    }
#endif
}

void CloudViewer::setBackfaceCulling(bool enabled, bool frontfaceCulling)
{
    _aBackfaceCulling->setChecked(enabled);
    _frontfaceCulling = frontfaceCulling;

    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    for(pcl::visualization::CloudActorMap::iterator iter=cloudActorMap->begin(); iter!=cloudActorMap->end(); ++iter)
    {
        iter->second.actor->GetProperty()->SetBackfaceCulling(_aBackfaceCulling->isChecked());
        iter->second.actor->GetProperty()->SetFrontfaceCulling(_frontfaceCulling);
    }
    this->update();
}

void CloudViewer::setPolygonPicking(bool enabled)
{
    _aPolygonPicking->setChecked(enabled);

    if(!_aPolygonPicking->isChecked())
    {
        vtkSmartPointer<vtkPointPicker> pp = vtkSmartPointer<vtkPointPicker>::New ();
        pp->SetTolerance (pp->GetTolerance());
        this->GetInteractor()->SetPicker (pp);
        setMouseTracking(false);
    }
    else
    {
        vtkSmartPointer<CloudViewerCellPicker> pp = vtkSmartPointer<CloudViewerCellPicker>::New ();
        pp->SetTolerance (pp->GetTolerance());
        this->GetInteractor()->SetPicker (pp);
        setMouseTracking(true);
    }
}

void CloudViewer::setRenderingRate(double rate)
{
    _renderingRate = rate;
    _visualizer->getInteractorStyle()->GetInteractor()->SetDesiredUpdateRate(_renderingRate);
}

void CloudViewer::setLighting(bool on)
{
    _aSetLighting->setChecked(on);
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    for(pcl::visualization::CloudActorMap::iterator iter=cloudActorMap->begin(); iter!=cloudActorMap->end(); ++iter)
    {
        iter->second.actor->GetProperty()->SetLighting(_aSetLighting->isChecked());
    }
    this->update();
}

void CloudViewer::setShading(bool on)
{
    _aSetFlatShading->setChecked(on);
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    for(pcl::visualization::CloudActorMap::iterator iter=cloudActorMap->begin(); iter!=cloudActorMap->end(); ++iter)
    {
        iter->second.actor->GetProperty()->SetInterpolation(_aSetFlatShading->isChecked()?VTK_FLAT:VTK_PHONG); // VTK_FLAT - VTK_GOURAUD - VTK_PHONG
    }
    this->update();
}

void CloudViewer::setEdgeVisibility(bool visible)
{
    _aSetEdgeVisibility->setChecked(visible);
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    for(pcl::visualization::CloudActorMap::iterator iter=cloudActorMap->begin(); iter!=cloudActorMap->end(); ++iter)
    {
        iter->second.actor->GetProperty()->SetEdgeVisibility(_aSetEdgeVisibility->isChecked());
    }
    this->update();
}

void CloudViewer::setInteractorLayer(int layer)
{
    _visualizer->getRendererCollection()->InitTraversal ();
    vtkRenderer* renderer = nullptr;
    int i =0;
    while ((renderer = _visualizer->getRendererCollection()->GetNextItem ()) != nullptr)
    {
        if(i==layer)
        {
            _visualizer->getInteractorStyle()->SetDefaultRenderer(renderer);
            _visualizer->getInteractorStyle()->SetCurrentRenderer(renderer);
            return;
        }
        ++i;
    }
//    UWARN("Could not set layer %d to interactor (layers=%d).", layer, _visualizer->getRendererCollection()->GetNumberOfItems());
}

void CloudViewer::getCameraPosition(
        float & x, float & y, float & z,
        float & focalX, float & focalY, float & focalZ,
        float & upX, float & upY, float & upZ) const
{
    std::vector<pcl::visualization::Camera> cameras;
    _visualizer->getCameras(cameras);
    if(cameras.size())
    {
        x = cameras.begin()->pos[0];
        y = cameras.begin()->pos[1];
        z = cameras.begin()->pos[2];
        focalX = cameras.begin()->focal[0];
        focalY = cameras.begin()->focal[1];
        focalZ = cameras.begin()->focal[2];
        upX = cameras.begin()->view[0];
        upY = cameras.begin()->view[1];
        upZ = cameras.begin()->view[2];
    }
    else
    {
        //		UERROR("No camera set!?");
    }
}

void CloudViewer::setCameraPosition(
        float x, float y, float z,
        float focalX, float focalY, float focalZ,
        float upX, float upY, float upZ)
{
    _lastCameraOrientation= _lastCameraPose= cv::Vec3f(0,0,0);
    _visualizer->setCameraPosition(x,y,z, focalX,focalY,focalX, upX,upY,upZ, 1);
}

void CloudViewer::updateCameraTargetPosition(const Eigen::Matrix4f & pose)
{
    if(!pose.isZero())
    {
        Eigen::Affine3f m = affine3fFrom(pose);
        Eigen::Vector3f pos = m.translation();

        Eigen::Vector3f lastPos(0,0,0);
        if(_trajectory->size())
        {
            lastPos[0]=_trajectory->back().x;
            lastPos[1]=_trajectory->back().y;
            lastPos[2]=_trajectory->back().z;
        }

        _trajectory->push_back(pcl::PointXYZ(pos[0], pos[1], pos[2]));
        if(_maxTrajectorySize>0)
        {
            while(_trajectory->size() > _maxTrajectorySize)
            {
                _trajectory->erase(_trajectory->begin());
            }
        }
        if(_aShowTrajectory->isChecked())
        {
            _visualizer->removeShape("trajectory");
            pcl::PolygonMesh mesh;
            pcl::Vertices vertices;
            vertices.vertices.resize(_trajectory->size());
            for(unsigned int i=0; i<vertices.vertices.size(); ++i)
            {
                vertices.vertices[i] = i;
            }
            pcl::toPCLPointCloud2(*_trajectory, mesh.cloud);
            mesh.polygons.push_back(vertices);
            _visualizer->addPolylineFromPolygonMesh(mesh, "trajectory", 1);
        }

        if(pose != _lastPose || _lastPose.isZero())
        {
            if(_lastPose.isZero())
            {
                _lastPose.setIdentity();
            }

            std::vector<pcl::visualization::Camera> cameras;
            _visualizer->getCameras(cameras);

            if(_aLockCamera->isChecked() || _aCameraOrtho->isChecked())
            {
                //update camera position
                Eigen::Vector3f lastPosePt = pointFrom(_lastPose);
                Eigen::Vector3f diff = pos - Eigen::Vector3f(lastPosePt.x(), lastPosePt.y(), lastPosePt.z());
                cameras.front().pos[0] += diff[0];
                cameras.front().pos[1] += diff[1];
                cameras.front().pos[2] += diff[2];
                cameras.front().focal[0] += diff[0];
                cameras.front().focal[1] += diff[1];
                cameras.front().focal[2] += diff[2];
            }
            else if(_aFollowCamera->isChecked())
            {
                Eigen::Vector3f vPosToFocal = Eigen::Vector3f(cameras.front().focal[0] - cameras.front().pos[0],
                        cameras.front().focal[1] - cameras.front().pos[1],
                        cameras.front().focal[2] - cameras.front().pos[2]).normalized();
                Eigen::Vector3f zAxis(cameras.front().view[0], cameras.front().view[1], cameras.front().view[2]);
                Eigen::Vector3f yAxis = zAxis.cross(vPosToFocal);
                Eigen::Vector3f xAxis = yAxis.cross(zAxis);
                Eigen::Matrix4f PR;
                PR << xAxis[0], xAxis[1], xAxis[2],0,
                        yAxis[0], yAxis[1], yAxis[2],0,
                        zAxis[0], zAxis[1], zAxis[2],0,
                        0, 0, 0, 1;

                PR = normalizeRotation(PR);

                Eigen::Matrix4f P;
                P << PR(0), PR(1), PR(2), cameras.front().pos[0],
                     PR(4), PR(5), PR(6), cameras.front().pos[1],
                     PR(8), PR(9), PR(10), cameras.front().pos[2],
                     0, 0, 0, 1;
                Eigen::Matrix4f F;
                F << PR(0), PR(1), PR(2), cameras.front().pos[0],
                     PR(4), PR(5), PR(6), cameras.front().pos[1],
                     PR(8), PR(9), PR(10), cameras.front().pos[2],
                     0, 0, 0, 1;
                Eigen::Matrix4f N = pose;
                Eigen::Matrix4f O = _lastPose;
                Eigen::Matrix4f O2N = O.inverse()*N;
                Eigen::Matrix4f F2O = F.inverse()*O;
                Eigen::Matrix4f T = F2O * O2N * F2O.inverse();
                Eigen::Matrix4f Fp = F * T;
                Eigen::Matrix4f P2F = P.inverse()*F;
                Eigen::Matrix4f Pp = P * P2F * T * P2F.inverse();

                Eigen::Vector3f PpPt = pointFrom(Pp);
                Eigen::Vector3f FpPt = pointFrom(Fp);

                cameras.front().pos[0] = PpPt.x();
                cameras.front().pos[1] = PpPt.y();
                cameras.front().pos[2] = PpPt.z();
                cameras.front().focal[0] = FpPt.x();
                cameras.front().focal[1] = FpPt.y();
                cameras.front().focal[2] = FpPt.z();
                //FIXME: the view up is not set properly...
                cameras.front().view[0] = _aLockViewZ->isChecked()?0:Fp(8);
                cameras.front().view[1] = _aLockViewZ->isChecked()?0:Fp(9);
                cameras.front().view[2] = _aLockViewZ->isChecked()?1:Fp(10);
            }

#if PCL_VERSION_COMPARE(>=, 1, 7, 2)
            if(_coordinates.find("reference") != _coordinates.end())
            {
                this->updateCoordinatePose("reference", pose);
            }
            else
#endif
            {
                this->addOrUpdateCoordinate("reference", pose, 0.2);
            }

            _visualizer->setCameraPosition(
                    cameras.front().pos[0], cameras.front().pos[1], cameras.front().pos[2],
                    cameras.front().focal[0], cameras.front().focal[1], cameras.front().focal[2],
                    cameras.front().view[0], cameras.front().view[1], cameras.front().view[2], 1);
        }
    }

    _lastPose = pose;
}

const QColor & CloudViewer::getDefaultBackgroundColor() const
{
    return _defaultBgColor;
}

void CloudViewer::setDefaultBackgroundColor(const QColor & color)
{
    if(_currentBgColor == _defaultBgColor)
    {
        setBackgroundColor(color);
    }
    _defaultBgColor = color;
}

const QColor & CloudViewer::getBackgroundColor() const
{
    return _currentBgColor;
}

void CloudViewer::setBackgroundColor(const QColor & color)
{
    _currentBgColor = color;
    _visualizer->setBackgroundColor(color.redF(), color.greenF(), color.blueF());
}

void CloudViewer::setCloudVisibility(const QString &id, bool isVisible)
{
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    pcl::visualization::CloudActorMap::iterator iter = cloudActorMap->find(id.toStdString());
    if(iter != cloudActorMap->end())
    {
        iter->second.actor->SetVisibility(isVisible?1:0);

        iter = cloudActorMap->find((id+"-normals").toStdString());
        if(iter != cloudActorMap->end())
        {
            iter->second.actor->SetVisibility(isVisible&&_aShowNormals->isChecked()?1:0);
        }
    }
    else
    {
        //		UERROR("Cannot find actor named \"%s\".", id.c_str());
    }
}

bool CloudViewer::getCloudVisibility(const QString & id)
{
    pcl::visualization::CloudActorMapPtr cloudActorMap = _visualizer->getCloudActorMap();
    pcl::visualization::CloudActorMap::iterator iter = cloudActorMap->find(id.toStdString());
    if(iter != cloudActorMap->end())
    {
        return iter->second.actor->GetVisibility() != 0;
    }
    else
    {
        //		UERROR("Cannot find actor named \"%s\".", id.c_str());
    }
    return false;
}

void CloudViewer::setCloudColorIndex(const QString &id, int index)
{
    if(index>0)
    {
        _visualizer->updateColorHandlerIndex(id.toStdString(), index-1);
    }
}

void CloudViewer::setCloudOpacity(const QString &id, double opacity)
{
    double lastOpacity;
    _visualizer->getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, lastOpacity, id.toStdString());
    if(lastOpacity != opacity)
    {
        _visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id.toStdString());
    }
}

void CloudViewer::setCloudPointSize(const QString &id, int size)
{
    double lastSize;
    _visualizer->getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, lastSize, id.toStdString());
    if((int)lastSize != size)
    {
        _visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, (double)size, id.toStdString());
    }
}

void CloudViewer::setCameraTargetLocked(bool enabled)
{
    _aLockCamera->setChecked(enabled);
}

void CloudViewer::setCameraTargetFollow(bool enabled)
{
    _aFollowCamera->setChecked(enabled);
}

void CloudViewer::setCameraFree()
{
    _aLockCamera->setChecked(false);
    _aFollowCamera->setChecked(false);
}

void CloudViewer::setCameraLockZ(bool enabled)
{
    _lastCameraOrientation= _lastCameraPose = cv::Vec3f(0,0,0);
    _aLockViewZ->setChecked(enabled);
}
void CloudViewer::setCameraOrtho(bool enabled)
{
    _lastCameraOrientation= _lastCameraPose = cv::Vec3f(0,0,0);
    CloudViewerInteractorStyle * interactor = CloudViewerInteractorStyle::SafeDownCast(this->GetInteractor()->GetInteractorStyle());
    if(interactor)
    {
        interactor->setOrthoMode(enabled);
        this->update();
    }
    _aCameraOrtho->setChecked(enabled);
}
bool CloudViewer::isCameraTargetLocked() const
{
    return _aLockCamera->isChecked();
}
bool CloudViewer::isCameraTargetFollow() const
{
    return _aFollowCamera->isChecked();
}
bool CloudViewer::isCameraFree() const
{
    return !_aFollowCamera->isChecked() && !_aLockCamera->isChecked();
}
bool CloudViewer::isCameraLockZ() const
{
    return _aLockViewZ->isChecked();
}
bool CloudViewer::isCameraOrtho() const
{
    return _aCameraOrtho->isChecked();
}
bool CloudViewer::isBackfaceCulling() const
{
    return _aBackfaceCulling->isChecked();
}
bool CloudViewer::isFrontfaceCulling() const
{
    return _frontfaceCulling;
}
bool CloudViewer::isPolygonPicking() const
{
    return _aPolygonPicking->isChecked();
}
bool CloudViewer::isLightingOn() const
{
    return _aSetLighting->isChecked();
}
bool CloudViewer::isShadingOn() const
{
    return _aSetFlatShading->isChecked();
}
bool CloudViewer::isEdgeVisible() const
{
    return _aSetEdgeVisibility->isChecked();
}
double CloudViewer::getRenderingRate() const
{
    return _renderingRate;
}

void CloudViewer::setGridShown(bool shown)
{
    _aShowGrid->setChecked(shown);
    if(shown)
    {
        this->addGrid();
    }
    else
    {
        this->removeGrid();
    }
}
bool CloudViewer::isGridShown() const
{
    return _aShowGrid->isChecked();
}
unsigned int CloudViewer::getGridCellCount() const
{
    return _gridCellCount;
}
float CloudViewer::getGridCellSize() const
{
    return _gridCellSize;
}
void CloudViewer::setGridCellCount(unsigned int count)
{
    if(count > 0)
    {
        _gridCellCount = count;
        if(_aShowGrid->isChecked())
        {
            this->removeGrid();
            this->addGrid();
        }
    }
    else
    {
        //		UERROR("Cannot set grid cell count < 1, count=%d", count);
    }
}
void CloudViewer::setGridCellSize(float size)
{
    if(size > 0)
    {
        _gridCellSize = size;
        if(_aShowGrid->isChecked())
        {
            this->removeGrid();
            this->addGrid();
        }
    }
    else
    {
        //		UERROR("Cannot set grid cell size <= 0, value=%f", size);
    }
}
void CloudViewer::addGrid()
{
    if(_gridLines.empty())
    {
        float cellSize = _gridCellSize;
        int cellCount = _gridCellCount;
        double r=0.5;
        double g=0.5;
        double b=0.5;
        int id = 0;
        float min = -float(cellCount/2) * cellSize;
        float max = float(cellCount/2) * cellSize;
        QString name;
        for(float i=min; i<=max; i += cellSize)
        {
            //over x
            name = QString("line%1").arg(++id);
            _visualizer->addLine(pcl::PointXYZ(i, min, 0.0f), pcl::PointXYZ(i, max, 0.0f), r, g, b, name.toStdString(), 1);
            _gridLines.push_back(name);
            //over y or z
            name = QString("line%d").arg(++id);
            _visualizer->addLine(
                        pcl::PointXYZ(min, i, 0),
                        pcl::PointXYZ(max, i, 0),
                        r, g, b, name.toStdString(), 1);
            _gridLines.push_back(name);
        }
    }
}

void CloudViewer::removeGrid()
{
    for(QStringList::iterator iter = _gridLines.begin(); iter!=_gridLines.end(); ++iter)
    {
        _visualizer->removeShape(iter->toStdString());
    }
    _gridLines.clear();
}

void CloudViewer::setNormalsShown(bool shown)
{
    _aShowNormals->setChecked(shown);
    QStringList ids = _addedClouds.keys();
    for(QStringList::iterator iter = ids.begin(); iter!=ids.end(); ++iter)
    {
        QString idNormals = *iter + "-normals";
        if(_addedClouds.find(idNormals) != _addedClouds.end())
        {
            this->setCloudVisibility(idNormals, this->getCloudVisibility(*iter) && shown);
        }
    }
}
bool CloudViewer::isNormalsShown() const
{
    return _aShowNormals->isChecked();
}
int CloudViewer::getNormalsStep() const
{
    return _normalsStep;
}
float CloudViewer::getNormalsScale() const
{
    return _normalsScale;
}
void CloudViewer::setNormalsStep(int step)
{
    if(step > 0)
    {
        _normalsStep = step;
    }
    else
    {
        //		UERROR("Cannot set normals step <= 0, step=%d", step);
    }
}
void CloudViewer::setNormalsScale(float scale)
{
    if(scale > 0)
    {
        _normalsScale= scale;
    }
    else
    {
        //		UERROR("Cannot set normals scale <= 0, value=%f", scale);
    }
}

void CloudViewer::buildPickingLocator(bool enable)
{
    _buildLocator = enable;
}

Eigen::Vector3f rotatePointAroundAxe(
        const Eigen::Vector3f & point,
        const Eigen::Vector3f & axis,
        float angle)
{
    Eigen::Vector3f direction = point;
    Eigen::Vector3f zAxis = axis;
    float dotProdZ = zAxis.dot(direction);
    Eigen::Vector3f ptOnZaxis = zAxis * dotProdZ;
    direction -= ptOnZaxis;
    Eigen::Vector3f xAxis = direction.normalized();
    Eigen::Vector3f yAxis = zAxis.cross(xAxis);

    Eigen::Matrix3f newFrame;
    newFrame << xAxis[0], yAxis[0], zAxis[0],
            xAxis[1], yAxis[1], zAxis[1],
            xAxis[2], yAxis[2], zAxis[2];

    // transform to axe frame
    // transpose=inverse for orthogonal matrices
    Eigen::Vector3f newDirection = newFrame.transpose() * direction;

    // rotate about z
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);
    float magnitude = newDirection.norm();
    newDirection[0] = ( magnitude * cosTheta );
    newDirection[1] = ( magnitude * sinTheta );

    // transform back to global frame
    direction = newFrame * newDirection;

    return direction + ptOnZaxis;
}

void CloudViewer::keyReleaseEvent(QKeyEvent * event) {
    if(event->key() == Qt::Key_Up ||
            event->key() == Qt::Key_Down ||
            event->key() == Qt::Key_Left ||
            event->key() == Qt::Key_Right)
    {
        _keysPressed -= (Qt::Key)event->key();
    }
    else
    {
        QVTKOpenGLWidget::keyPressEvent(event);
    }
}

void CloudViewer::keyPressEvent(QKeyEvent * event)
{
    if(event->key() == Qt::Key_Up ||
            event->key() == Qt::Key_Down ||
            event->key() == Qt::Key_Left ||
            event->key() == Qt::Key_Right)
    {
        _keysPressed += (Qt::Key)event->key();

        std::vector<pcl::visualization::Camera> cameras;
        _visualizer->getCameras(cameras);

        //update camera position
        Eigen::Vector3f pos(cameras.front().pos[0], cameras.front().pos[1], _aLockViewZ->isChecked()?0:cameras.front().pos[2]);
        Eigen::Vector3f focal(cameras.front().focal[0], cameras.front().focal[1], _aLockViewZ->isChecked()?0:cameras.front().focal[2]);
        Eigen::Vector3f viewUp(cameras.front().view[0], cameras.front().view[1], cameras.front().view[2]);
        Eigen::Vector3f cummulatedDir(0,0,0);
        Eigen::Vector3f cummulatedFocalDir(0,0,0);
        float step = 0.2f;
        float stepRot = 0.02f; // radian
        if(_keysPressed.contains(Qt::Key_Up))
        {
            Eigen::Vector3f dir;
            if(event->modifiers() & Qt::ShiftModifier)
            {
                dir = viewUp * step;// up
            }
            else
            {
                dir = (focal-pos).normalized() * step; // forward
            }
            cummulatedDir += dir;
        }
        if(_keysPressed.contains(Qt::Key_Down))
        {
            Eigen::Vector3f dir;
            if(event->modifiers() & Qt::ShiftModifier)
            {
                dir = viewUp * -step;// down
            }
            else
            {
                dir = (focal-pos).normalized() * -step; // backward
            }
            cummulatedDir += dir;
        }
        if(_keysPressed.contains(Qt::Key_Right))
        {
            if(event->modifiers() & Qt::ShiftModifier)
            {
                // rotate right
                Eigen::Vector3f point = (focal-pos);
                Eigen::Vector3f newPoint = rotatePointAroundAxe(point, viewUp, -stepRot);
                Eigen::Vector3f diff = newPoint - point;
                cummulatedFocalDir += diff;
            }
            else
            {
                Eigen::Vector3f dir = ((focal-pos).cross(viewUp)).normalized() * step; // strafing right
                cummulatedDir += dir;
            }
        }
        if(_keysPressed.contains(Qt::Key_Left))
        {
            if(event->modifiers() & Qt::ShiftModifier)
            {
                // rotate left
                Eigen::Vector3f point = (focal-pos);
                Eigen::Vector3f newPoint = rotatePointAroundAxe(point, viewUp, stepRot);
                Eigen::Vector3f diff = newPoint - point;
                cummulatedFocalDir += diff;
            }
            else
            {
                Eigen::Vector3f dir = ((focal-pos).cross(viewUp)).normalized() * -step; // strafing left
                cummulatedDir += dir;
            }
        }

        cameras.front().pos[0] += cummulatedDir[0];
        cameras.front().pos[1] += cummulatedDir[1];
        cameras.front().pos[2] += cummulatedDir[2];
        cameras.front().focal[0] += cummulatedDir[0] + cummulatedFocalDir[0];
        cameras.front().focal[1] += cummulatedDir[1] + cummulatedFocalDir[1];
        cameras.front().focal[2] += cummulatedDir[2] + cummulatedFocalDir[2];
        _visualizer->setCameraPosition(
                    cameras.front().pos[0], cameras.front().pos[1], cameras.front().pos[2],
                cameras.front().focal[0], cameras.front().focal[1], cameras.front().focal[2],
                cameras.front().view[0], cameras.front().view[1], cameras.front().view[2], 1);

        update();

        Q_EMIT configChanged();
    }
    else
    {
        QVTKOpenGLWidget::keyPressEvent(event);
    }
}

void CloudViewer::mousePressEvent(QMouseEvent * event)
{
    if(event->button() == Qt::RightButton)
    {
        event->accept();
    }
    else
    {
        QVTKOpenGLWidget::mousePressEvent(event);
    }
}

void CloudViewer::mouseMoveEvent(QMouseEvent * event)
{
    QVTKOpenGLWidget::mouseMoveEvent(event);

    // camera view up z locked?
    if(_aLockViewZ->isChecked() && !_aCameraOrtho->isChecked())
    {
        std::vector<pcl::visualization::Camera> cameras;
        _visualizer->getCameras(cameras);

        cv::Vec3d newCameraOrientation = cv::Vec3d(0,0,1).cross(cv::Vec3d(cameras.front().pos)-cv::Vec3d(cameras.front().focal));

        if(	_lastCameraOrientation!=cv::Vec3d(0,0,0) &&
                _lastCameraPose!=cv::Vec3d(0,0,0) &&
                (sign(_lastCameraOrientation[0]) != sign(newCameraOrientation[0]) &&
                 sign(_lastCameraOrientation[1]) != sign(newCameraOrientation[1])))
        {
            cameras.front().pos[0] = _lastCameraPose[0];
            cameras.front().pos[1] = _lastCameraPose[1];
            cameras.front().pos[2] = _lastCameraPose[2];
        }
        else if(newCameraOrientation != cv::Vec3d(0,0,0))
        {
            _lastCameraOrientation = newCameraOrientation;
            _lastCameraPose = cv::Vec3d(cameras.front().pos);
        }
        else
        {
            if(cameras.front().view[2] == 0)
            {
                cameras.front().pos[0] -= 0.00001*cameras.front().view[0];
                cameras.front().pos[1] -= 0.00001*cameras.front().view[1];
            }
            else
            {
                cameras.front().pos[0] -= 0.00001;
            }
        }
        cameras.front().view[0] = 0;
        cameras.front().view[1] = 0;
        cameras.front().view[2] = 1;

        _visualizer->setCameraPosition(
                    cameras.front().pos[0], cameras.front().pos[1], cameras.front().pos[2],
                cameras.front().focal[0], cameras.front().focal[1], cameras.front().focal[2],
                cameras.front().view[0], cameras.front().view[1], cameras.front().view[2], 1);

    }
    this->update();

    Q_EMIT configChanged();
}

void CloudViewer::wheelEvent(QWheelEvent * event)
{
    QVTKOpenGLWidget::wheelEvent(event);
    if(_aLockViewZ->isChecked() && !_aCameraOrtho->isChecked())
    {
        std::vector<pcl::visualization::Camera> cameras;
        _visualizer->getCameras(cameras);
        _lastCameraPose = cv::Vec3d(cameras.front().pos);
    }
    Q_EMIT configChanged();
}

void CloudViewer::contextMenuEvent(QContextMenuEvent * event)
{
    QAction * a = _menu->exec(event->globalPos());
    if(a)
    {
        handleAction(a);
        Q_EMIT configChanged();
    }
}

void CloudViewer::handleAction(QAction * a)
{
    if(a == _aSetTrajectorySize)
    {
        bool ok;
        int value = QInputDialog::getInt(this, tr("Set trajectory size"), tr("Size (0=infinite)"), _maxTrajectorySize, 0, 10000, 10, &ok);
        if(ok)
        {
            _maxTrajectorySize = value;
        }
    }
    else if(a == _aClearTrajectory)
    {
        this->clearTrajectory();
    }
    else if(a == _aShowFrustum)
    {
        this->setFrustumShown(a->isChecked());
    }
    else if(a == _aSetFrustumScale)
    {
        bool ok;
        double value = QInputDialog::getDouble(this, tr("Set frustum scale"), tr("Scale"), _frustumScale, 0.0, 999.0, 1, &ok);
        if(ok)
        {
            this->setFrustumScale(value);
        }
    }
    else if(a == _aSetFrustumColor)
    {
        QColor value = QColorDialog::getColor(_frustumColor, this);
        if(value.isValid())
        {
            this->setFrustumColor(value);
        }
    }
    else if(a == _aResetCamera)
    {
        this->resetCamera();
    }
    else if(a == _aShowGrid)
    {
        if(_aShowGrid->isChecked())
        {
            this->addGrid();
        }
        else
        {
            this->removeGrid();
        }

        this->update();
    }
    else if(a == _aSetGridCellCount)
    {
        bool ok;
        int value = QInputDialog::getInt(this, tr("Set grid cell count"), tr("Count"), _gridCellCount, 1, 10000, 10, &ok);
        if(ok)
        {
            this->setGridCellCount(value);
        }
    }
    else if(a == _aSetGridCellSize)
    {
        bool ok;
        double value = QInputDialog::getDouble(this, tr("Set grid cell size"), tr("Size (m)"), _gridCellSize, 0.01, 10, 2, &ok);
        if(ok)
        {
            this->setGridCellSize(value);
        }
    }
    else if(a == _aShowNormals)
    {
        this->setNormalsShown(_aShowNormals->isChecked());
        this->update();
    }
    else if(a == _aSetNormalsStep)
    {
        bool ok;
        int value = QInputDialog::getInt(this, tr("Set normals step"), tr("Step"), _normalsStep, 1, 10000, 1, &ok);
        if(ok)
        {
            this->setNormalsStep(value);
        }
    }
    else if(a == _aSetNormalsScale)
    {
        bool ok;
        double value = QInputDialog::getDouble(this, tr("Set normals scale"), tr("Scale (m)"), _normalsScale, 0.01, 10, 2, &ok);
        if(ok)
        {
            this->setNormalsScale(value);
        }
    }
    else if(a == _aSetBackgroundColor)
    {
        QColor color = this->getDefaultBackgroundColor();
        color = QColorDialog::getColor(color, this);
        if(color.isValid())
        {
            this->setDefaultBackgroundColor(color);
            this->update();
        }
    }
    else if(a == _aSetRenderingRate)
    {
        bool ok;
        double value = QInputDialog::getDouble(this, tr("Rendering rate"), tr("Rate (hz)"), _renderingRate, 0, 60, 0, &ok);
        if(ok)
        {
            this->setRenderingRate(value);
        }
    }
    else if(a == _aLockViewZ)
    {
        if(_aLockViewZ->isChecked())
        {
            this->update();
        }
    }
    else if(a == _aCameraOrtho)
    {
        this->setCameraOrtho(_aCameraOrtho->isChecked());
    }
    else if(a == _aSetLighting)
    {
        this->setLighting(_aSetLighting->isChecked());
    }
    else if(a == _aSetFlatShading)
    {
        this->setShading(_aSetFlatShading->isChecked());
    }
    else if(a == _aSetEdgeVisibility)
    {
        this->setEdgeVisibility(_aSetEdgeVisibility->isChecked());
    }
    else if(a == _aBackfaceCulling)
    {
        this->setBackfaceCulling(_aBackfaceCulling->isChecked(), _frontfaceCulling);
    }
    else if(a == _aPolygonPicking)
    {
        this->setPolygonPicking(_aPolygonPicking->isChecked());
    }
}

void CloudViewer::mouseDoubleClickEvent(QMouseEvent *event)
{
    if(event->button() == Qt::RightButton)
    {
        QAction * a = _menu->exec(event->globalPos());
        if(a)
        {
            handleAction(a);
            Q_EMIT configChanged();
        }
        event->accept();
    }
    else
    {
        QVTKOpenGLWidget::mouseDoubleClickEvent(event);
    }
}
