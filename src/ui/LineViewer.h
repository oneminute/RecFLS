#pragma once

#include <QMainWindow>
#include <QScopedPointer>
#include <QList>

#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QCameraLens>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>
#include <Qt3DInput/QInputAspect>
#include <Qt3DRender/QRenderAspect>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DExtras/QTorusMesh>
#include <Qt3DExtras/qt3dwindow.h>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QFirstPersonCameraController>
#include <Qt3DRender/QObjectPicker>
#include <Qt3DRender/QPickEvent>

#include <pcl/common/common.h>

#include "extractor/LineSegment.h"
#include <pcl/kdtree/kdtree_flann.h>

#define XAXIS QVector3D(1.0f, 0.0f, 0.0f)
#define YAXIS QVector3D(0.0f, 1.0f, 0.0f)
#define ZAXIS QVector3D(0.0f, 0.0f, 1.0f)

namespace Ui
{
    class LineViewer;
};

class LineViewer : public QMainWindow
{
    Q_OBJECT
public:
    LineViewer(QWidget* parent = nullptr);
    ~LineViewer();

    void createAxisAngleScene();
    void createAxisArrow(qreal angles, const QVector3D& axis, qreal distance, const QColor& color, Qt3DCore::QEntity* axisEntity);
    void createCone(const QVector3D& dir, const QVector3D& pos, const QColor& color, Qt3DCore::QEntity* parentEntity);
    Qt3DCore::QEntity* createAxis(const QVector3D& pos, qreal length, Qt3DCore::QEntity* parentEntity);
    Qt3DCore::QEntity* createLine(const QVector3D& start, const QVector3D& end, const QColor& color, Qt3DCore::QEntity* parentEntity, bool showArrow = true);
    Qt3DCore::QEntity* createLine(const QVector3D& start, const QVector3D& end, const QColor& color, qreal width, Qt3DCore::QEntity* parentEntity);

    void resetCamera();

    template<typename T>
    static T* getComponent(Qt3DCore::QEntity* entity);

    void setEntityColor(Qt3DCore::QEntity* entity, QColor color);

public slots:
    void updateTransform(Qt3DCore::QTransform* transform, qreal pitch, qreal yaw, qreal roll, qreal x, qreal y, qreal z, bool abstract);
    void randomGenerate();
    void arrangedGenerate();
    void clearLines();
    void searchLine();

protected:
    virtual void closeEvent(QCloseEvent* event);

protected slots:
    void onTransformValuesChanged(double value = 0);
    void onObjectPicked(Qt3DRender::QPickEvent* pick);

private:
    QScopedPointer<Ui::LineViewer> m_ui;
    Qt3DExtras::Qt3DWindow* m_canvas;
    Qt3DCore::QEntity* m_rootEntity;
    Qt3DExtras::QOrbitCameraController* m_camController;

    Qt3DCore::QEntity* m_currentObject;
    Qt3DCore::QEntity* m_foundLine;

    pcl::PointCloud<LineSegment>::Ptr m_linesCloud;
    pcl::KdTreeFLANN<LineSegment>::Ptr m_linesTree;
    QList<Qt3DCore::QEntity*> m_lines;
    LineSegment m_line;
};

template<typename T>
inline T* LineViewer::getComponent(Qt3DCore::QEntity* entity)
{
	QVector<T*> components = entity->componentsOfType<T>();
	if (components.isEmpty())
		return nullptr;
	return components[0];
}
