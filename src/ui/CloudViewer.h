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

#ifndef CLOUDVIEWER_H_
#define CLOUDVIEWER_H_

#include <QVTKOpenGLWidget.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/mouse_event.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/visualization/interactor_style.h>
#include <pcl/point_types.h>

#include <QtCore/QMap>
#include <QtCore/QSet>
#include <QtCore/qnamespace.h>
#include <QtCore/QSettings>

#include <opencv2/opencv.hpp>
#include <set>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace pcl {
	namespace visualization {
		class PCLVisualizer;
	}
}

class QMenu;
class vtkProp;
template <typename T> class vtkSmartPointer;
class vtkOBBTree;

class CloudViewer;

class CloudViewerInteractorStyle: public pcl::visualization::PCLVisualizerInteractorStyle
{
public:
    static CloudViewerInteractorStyle *New ();
    vtkTypeMacro(CloudViewerInteractorStyle, pcl::visualization::PCLVisualizerInteractorStyle)

public:
    CloudViewerInteractorStyle();
    virtual void Rotate() override;
    void setOrthoMode(bool enabled);
protected:
    virtual void OnMouseMove() override;
    virtual void OnLeftButtonDown() override;

protected:
    friend class CloudViewer;
    void setCloudViewer(CloudViewer * cloudViewer) {viewer_ = cloudViewer;}
    CloudViewer * viewer_;

private:
    unsigned int NumberOfClicks;
    int PreviousPosition[2];
    int ResetPixelDistance;
    float PreviousMeasure[3];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsHolder_;
    bool orthoMode_;
};

class CloudViewer : public QVTKOpenGLWidget
{
	Q_OBJECT

public:
    CloudViewer(QWidget * parent = nullptr, CloudViewerInteractorStyle* style = CloudViewerInteractorStyle::New());
	virtual ~CloudViewer();

	void saveSettings(QSettings & settings, const QString & group = "") const;
	void loadSettings(QSettings & settings, const QString & group = "");

	bool updateCloudPose(
        const QString & id,
        const Eigen::Matrix4f & pose); //including mesh

	bool addCloud(
            const QString & id,
			const pcl::PCLPointCloud2Ptr & binaryCloud,
            const Eigen::Matrix4f & pose,
			bool rgb,
			bool hasNormals,
			bool hasIntensity,
			const QColor & color = QColor());

	bool addCloud(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & cloud,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
			const QColor & color = QColor());

	bool addCloud(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
			const QColor & color = QColor());

	bool addCloud(
                const QString & id,
				const pcl::PointCloud<pcl::PointXYZINormal>::Ptr & cloud,
                const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
				const QColor & color = QColor());

	bool addCloud(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
			const QColor & color = QColor());

	bool addCloud(
            const QString & id,
			const pcl::PointCloud<pcl::PointNormal>::Ptr & cloud,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
			const QColor & color = QColor());

	bool addCloud(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity(),
			const QColor & color = QColor());

	bool addCloudMesh(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
            const QVector<pcl::Vertices> & polygons,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity());

	bool addCloudMesh(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
            const QVector<pcl::Vertices> & polygons,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity());

	bool addCloudMesh(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & cloud,
            const QVector<pcl::Vertices> & polygons,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity());

	bool addCloudMesh(
            const QString & id,
			const pcl::PolygonMesh::Ptr & mesh,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity());

	// Only one texture per mesh is supported!
	bool addCloudTextureMesh(
            const QString & id,
			const pcl::TextureMesh::Ptr & textureMesh,
			const cv::Mat & texture,
            const Eigen::Matrix4f & pose = Eigen::Matrix4f::Identity());

	// Only one texture per mesh is supported!
	bool addTextureMesh (
		   const pcl::TextureMesh &mesh,
		   const cv::Mat & texture, 
           const QString &id = "texture",
		   int viewport = 0);
	bool addOccupancyGridMap(
			const cv::Mat & map8U,
			float resolution, // cell size
			float xMin,
			float yMin,
			float opacity);
	void removeOccupancyGridMap();

	void updateCameraTargetPosition(
        const Eigen::Matrix4f & pose);

	void addOrUpdateCoordinate(
            const QString & id,
            const Eigen::Matrix4f & transform,
			double scale,
			bool foreground=false);
	bool updateCoordinatePose(
            const QString & id,
            const Eigen::Matrix4f & transform);
    void removeCoordinate(const QString & id);
    void removeAllCoordinates(const QString & prefix = "");
    const QSet<QString> & getAddedCoordinates() const {return _coordinates;}

    void addOrUpdateLine(const QString & id,
                const Eigen::Vector3f &from,
                const Eigen::Vector3f &to,
                const QColor & color,
                bool arrow = false,
                bool foreground = false);
    void removeLine(const QString & id);
	void removeAllLines();
    const QSet<QString> & getAddedLines() const {return _lines;}

    void addOrUpdateSphere(const QString & id,
                const Eigen::Vector3f &pose,
                float radius,
                const QColor & color,
                bool foreground = false);
    void removeSphere(const QString & id);
	void removeAllSpheres();
    const QSet<QString> & getAddedSpheres() const {return _spheres;}

    void addOrUpdateCube(const QString & id,
                const Eigen::Matrix4f &pose, // center of the cube
                float width,  // e.g., along x axis
                float height, // e.g., along y axis
                float depth,  // e.g., along z axis
                const QColor & color,
                bool wireframe = false,
                bool foreground = false);
    void removeCube(const QString & id);
	void removeAllCubes();
    const QSet<QString> & getAddedCubes() const {return _cubes;}

	void addOrUpdateQuad(
            const QString & id,
            const Eigen::Matrix4f & pose,
			float width,
			float height,
			const QColor & color,
			bool foreground = false);
	void addOrUpdateQuad(
            const QString & id,
            const Eigen::Matrix4f & pose,
			float widthLeft,
			float widthRight,
			float heightBottom,
			float heightTop,
			const QColor & color,
			bool foreground = false);
    void removeQuad(const QString & id);
	void removeAllQuads();
    const QSet<QString> & getAddedQuads() const {return _quads;}

	void addOrUpdateFrustum(
            const QString & id,
            const Eigen::Matrix4f & transform,
            const Eigen::Matrix4f & localTransform,
			double scale,
			const QColor & color = QColor());
	bool updateFrustumPose(
            const QString & id,
            const Eigen::Matrix4f & pose);
    void removeFrustum(const QString &id);
	void removeAllFrustums(bool exceptCameraReference = false);
    const QMap<QString, Eigen::Matrix4f> & getAddedFrustums() const {return _frustums;}

	void addOrUpdateGraph(
            const QString & id,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & graph,
			const QColor & color = Qt::gray);
    void removeGraph(const QString & id);
	void removeAllGraphs();

    void addOrUpdateText(const QString & id,
            const QString & text,
            const Eigen::Vector3f &position,
            double scale,
            const QColor & color,
            bool foreground = true);
    void removeText(const QString & id);
	void removeAllTexts();
    const QSet<QString> & getAddedTexts() const {return _texts;}

	bool isTrajectoryShown() const;
	unsigned int getTrajectorySize() const;
	void setTrajectoryShown(bool shown);
	void setTrajectorySize(unsigned int value);
	void clearTrajectory();
	bool isFrustumShown() const;
	float getFrustumScale() const;
	QColor getFrustumColor() const;
	void setFrustumShown(bool shown);
	void setFrustumScale(float value);
	void setFrustumColor(QColor value);
	void resetCamera();

	void removeAllClouds(); //including meshes
    bool removeCloud(const QString & id); //including mesh

    bool getPose(const QString & id, Eigen::Matrix4f & pose); //including meshes
    bool getCloudVisibility(const QString & id);

    const QMap<QString, Eigen::Matrix4f> & getAddedClouds() const {return _addedClouds;} //including meshes
	const QColor & getDefaultBackgroundColor() const;
	const QColor & getBackgroundColor() const;
    Eigen::Matrix4f getTargetPose() const;
    QString getIdByActor(vtkProp * actor) const;
    QColor getColor(const QString & id);
    void setColor(const QString & id, const QColor & color);

	void setBackfaceCulling(bool enabled, bool frontfaceCulling);
	void setPolygonPicking(bool enabled);
	void setRenderingRate(double rate);
	void setLighting(bool on);
	void setShading(bool on);
	void setEdgeVisibility(bool visible);
	void setInteractorLayer(int layer);

	bool isBackfaceCulling() const;
	bool isFrontfaceCulling() const;
	bool isPolygonPicking() const;
	bool isLightingOn() const;
	bool isShadingOn() const;
	bool isEdgeVisible() const;
	double getRenderingRate() const;

	void getCameraPosition(
			float & x, float & y, float & z,
			float & focalX, float & focalY, float & focalZ,
			float & upX, float & upY, float & upZ) const;
	bool isCameraTargetLocked() const;
	bool isCameraTargetFollow() const;
	bool isCameraFree() const;
	bool isCameraLockZ() const;
	bool isCameraOrtho() const;
	bool isGridShown() const;
	unsigned int getGridCellCount() const;
	float getGridCellSize() const;

	void setCameraPosition(
			float x, float y, float z,
			float focalX, float focalY, float focalZ,
			float upX, float upY, float upZ);
	void setCameraTargetLocked(bool enabled = true);
	void setCameraTargetFollow(bool enabled = true);
	void setCameraFree();
	void setCameraLockZ(bool enabled = true);
	void setCameraOrtho(bool enabled = true);
	void setGridShown(bool shown);
	void setNormalsShown(bool shown);
	void setGridCellCount(unsigned int count);
	void setGridCellSize(float size);
	bool isNormalsShown() const;
	int getNormalsStep() const;
	float getNormalsScale() const;
	void setNormalsStep(int step);
	void setNormalsScale(float scale);
	void buildPickingLocator(bool enable);
    const QMap<QString, vtkSmartPointer<vtkOBBTree>> & getLocators() const {return _locators;}

public Q_SLOTS:
	void setDefaultBackgroundColor(const QColor & color);
	void setBackgroundColor(const QColor & color);
    void setCloudVisibility(const QString & id, bool isVisible);
    void setCloudColorIndex(const QString & id, int index);
    void setCloudOpacity(const QString & id, double opacity = 1.0);
    void setCloudPointSize(const QString & id, int size);
	virtual void clear();

Q_SIGNALS:
	void configChanged();

protected:
	virtual void keyReleaseEvent(QKeyEvent * event);
	virtual void keyPressEvent(QKeyEvent * event);
	virtual void mousePressEvent(QMouseEvent * event);
	virtual void mouseMoveEvent(QMouseEvent * event);
	virtual void wheelEvent(QWheelEvent * event);
	virtual void contextMenuEvent(QContextMenuEvent * event);
	virtual void handleAction(QAction * event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    QMenu * menu() {return _menu;}
	pcl::visualization::PCLVisualizer * visualizer() {return _visualizer;}

private:
	void createMenu();
	void addGrid();
	void removeGrid();

private:
    pcl::visualization::PCLVisualizer * _visualizer;
    QAction * _aLockCamera;
    QAction * _aFollowCamera;
    QAction * _aResetCamera;
    QAction * _aLockViewZ;
    QAction * _aCameraOrtho;
    QAction * _aShowTrajectory;
    QAction * _aSetTrajectorySize;
    QAction * _aClearTrajectory;
    QAction * _aShowFrustum;
    QAction * _aSetFrustumScale;
    QAction * _aSetFrustumColor;
    QAction * _aShowGrid;
    QAction * _aSetGridCellCount;
    QAction * _aSetGridCellSize;
    QAction * _aShowNormals;
	QAction * _aSetNormalsStep;
	QAction * _aSetNormalsScale;
    QAction * _aSetBackgroundColor;
    QAction * _aSetRenderingRate;
    QAction * _aSetLighting;
    QAction * _aSetFlatShading;
    QAction * _aSetEdgeVisibility;
    QAction * _aBackfaceCulling;
    QAction * _aPolygonPicking;
    QMenu * _menu;
    QSet<QString> _graphes;
    QSet<QString> _coordinates;
    QSet<QString> _texts;
    QSet<QString> _lines;
    QSet<QString> _spheres;
    QSet<QString> _cubes;
    QSet<QString> _quads;
    QMap<QString, Eigen::Matrix4f> _frustums;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _trajectory;
    unsigned int _maxTrajectorySize;
    float _frustumScale;
    QColor _frustumColor;
    unsigned int _gridCellCount;
    float _gridCellSize;
    int _normalsStep;
    float _normalsScale;
    bool _buildLocator;
    QMap<QString, vtkSmartPointer<vtkOBBTree> > _locators;
    cv::Vec3d _lastCameraOrientation;
    cv::Vec3d _lastCameraPose;
    QMap<QString, Eigen::Matrix4f> _addedClouds; // include cloud, scan, meshes
    Eigen::Matrix4f _lastPose;
    QList<QString> _gridLines;
    QSet<Qt::Key> _keysPressed;
    QColor _defaultBgColor;
    QColor _currentBgColor;
    bool _frontfaceCulling;
    double _renderingRate;
    vtkProp * _octomapActor;

};

#endif /* CLOUDVIEWER_H_ */
