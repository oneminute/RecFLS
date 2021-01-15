#include "LineViewer.h"

#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QRandomGenerator>
#include <QQuaternion>
#include <QAction>

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QConeMesh>
#include <Qt3DRender/QAttribute>
#include <Qt3DRender/QBuffer>
#include <Qt3DRender/QGeometry>
#include <QtMath>

#include <ui/ui_LineViewer.h>
#include "util/Utils.h"

LineViewer::LineViewer(QWidget* parent)
	: QMainWindow(parent)
	, m_ui(new Ui::LineViewer)
	, m_canvas(nullptr)
    , m_rootEntity(nullptr)
	, m_currentObject(nullptr)
	, m_foundLine(nullptr)
	, m_linesCloud(new pcl::PointCloud<LineSegment>())
	, m_linesTree(new pcl::KdTreeFLANN<LineSegment>())
{
	m_ui->setupUi(this);
	m_canvas = new Qt3DExtras::Qt3DWindow();
	m_canvas->defaultFrameGraph()->setClearColor(QColor(QRgb(0x202020)));
	QWidget* container = QWidget::createWindowContainer(m_canvas);

	QVBoxLayout* layout = new QVBoxLayout(centralWidget());
	layout->addWidget(container);

	m_ui->toolButtonRandomGenerate->setDefaultAction(m_ui->actionRandomGenerate);
	m_ui->toolButtonArrangedGenerate->setDefaultAction(m_ui->actionArrangedGenerate);
	m_ui->toolButtonClearLines->setDefaultAction(m_ui->actionClearLines);

	connect(m_ui->doubleSpinBoxPitch, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->doubleSpinBoxYaw, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->doubleSpinBoxRoll, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->doubleSpinBoxTransX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->doubleSpinBoxTransY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->doubleSpinBoxTransZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &LineViewer::onTransformValuesChanged);
	connect(m_ui->actionRandomGenerate, &QAction::triggered, this, &LineViewer::randomGenerate);
	connect(m_ui->actionArrangedGenerate, &QAction::triggered, this, &LineViewer::arrangedGenerate);
	connect(m_ui->actionClearLines, &QAction::triggered, this, &LineViewer::clearLines);

    m_rootEntity = new Qt3DCore::QEntity;
	createAxis(QVector3D(0, 0, 0), 1.5f, m_rootEntity);

	m_linesTree->setPointRepresentation(pcl::PointRepresentation<LineSegment>::Ptr(new LocalDescriptorRepresentation));
}

LineViewer::~LineViewer()
{
}

void LineViewer::createAxisAngleScene()
{
	Qt3DCore::QEntity* line = createLine({ 0, 0, 0 }, { 0, 0, 1 }, Qt::yellow, m_rootEntity);
	// transform
	QMatrix4x4 matrix;
	matrix.setToIdentity();
	Qt3DCore::QTransform* transform = new Qt3DCore::QTransform;
	transform->setMatrix(matrix);
	line->addComponent(transform);
	line->setObjectName("line");

	Qt3DRender::QObjectPicker* picker = new Qt3DRender::QObjectPicker(m_rootEntity);
	line->addComponent(picker);
	connect(picker, &Qt3DRender::QObjectPicker::clicked, this, &LineViewer::onObjectPicked);
	m_currentObject = line;

	resetCamera();

	m_canvas->setRootEntity(m_rootEntity);
	onTransformValuesChanged();
}

void LineViewer::createAxisArrow(qreal angles, const QVector3D& axis, qreal distance, const QColor& color, Qt3DCore::QEntity* axisEntity)
{
	QMatrix4x4 matrix;
	matrix.setToIdentity();
	matrix.rotate(angles, axis);
	matrix.translate(YAXIS * distance);
	Qt3DCore::QTransform* trans = new Qt3DCore::QTransform(axisEntity);
	trans->setMatrix(matrix);

	Qt3DExtras::QPhongMaterial* mat = new Qt3DExtras::QPhongMaterial(axisEntity);
	mat->setAmbient(color);

	Qt3DExtras::QConeMesh* cone = new Qt3DExtras::QConeMesh(axisEntity);
	cone->setLength(0.25f);
	cone->setTopRadius(0.0f);
	cone->setBottomRadius(0.02f);

	Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(axisEntity);
	entity->addComponent(cone);
	entity->addComponent(mat);
	entity->addComponent(trans);
}

void LineViewer::createCone(const QVector3D& dir, const QVector3D& pos, const QColor& color, Qt3DCore::QEntity* parentEntity)
{
	Qt3DCore::QTransform* trans = new Qt3DCore::QTransform();
	Eigen::AngleAxisf aa = axisAnglesFrom2Vectors(YAXIS, dir);
	Eigen::Vector3f axis = aa.axis();
	trans->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(axis.x(), axis.y(), axis.z()), qRadiansToDegrees(aa.angle())));
	trans->setTranslation(pos);

	Qt3DExtras::QPhongMaterial* mat = new Qt3DExtras::QPhongMaterial();
	mat->setAmbient(color);

	Qt3DExtras::QConeMesh* cone = new Qt3DExtras::QConeMesh();
	cone->setLength(0.25f);
	cone->setTopRadius(0.0f);
	cone->setBottomRadius(0.02f);

	Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(parentEntity);
	entity->addComponent(cone);
	entity->addComponent(mat);
	entity->addComponent(trans);
}

Qt3DCore::QEntity* LineViewer::createAxis(const QVector3D& pos, qreal length, Qt3DCore::QEntity* parentEntity)
{
	Qt3DCore::QEntity* axisEntity = new Qt3DCore::QEntity(parentEntity);
	QVector3D xEnd = pos + XAXIS * length;
	QVector3D yEnd = pos + YAXIS * length;
	QVector3D zEnd = pos + ZAXIS * length;
	Qt3DCore::QEntity* xEntity = createLine(pos, xEnd, Qt::red, axisEntity, false);
	Qt3DCore::QEntity* yEntity = createLine(pos, yEnd, Qt::green, axisEntity, false);
	Qt3DCore::QEntity* zEntity = createLine(pos, zEnd, Qt::blue, axisEntity, false);

	createAxisArrow(-90, ZAXIS, length, Qt::red, axisEntity);
	createAxisArrow(0, YAXIS, length, Qt::green, axisEntity);
	createAxisArrow(90, XAXIS, length, Qt::blue, axisEntity);

	return axisEntity;
}

Qt3DCore::QEntity* LineViewer::createLine(const QVector3D& start, const QVector3D& end, const QColor& color, Qt3DCore::QEntity* parentEntity, bool showArrow)
{
	Qt3DRender::QGeometry* geometry = new Qt3DRender::QGeometry();
	QVector3D dir = end - start;

	// position vertices (start and end)
	QByteArray bufferBytes;
	bufferBytes.resize(3 * 2 * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
	float* positions = reinterpret_cast<float*>(bufferBytes.data());
	*positions++ = start.x();
	*positions++ = start.y();
	*positions++ = start.z();
	*positions++ = end.x();
	*positions++ = end.y();
	*positions++ = end.z();

	Qt3DRender::QBuffer* buf = new Qt3DRender::QBuffer(geometry);
	buf->setData(bufferBytes);

	Qt3DRender::QAttribute* positionAttribute = new Qt3DRender::QAttribute(geometry);
	positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());
	positionAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setByteStride(3 * sizeof(float));
	positionAttribute->setCount(2);
	geometry->addAttribute(positionAttribute); // We add the vertices in the geometry

	// connectivity between vertices
	QByteArray indexBytes;
	indexBytes.resize(2 * sizeof(unsigned int)); // start to end
	unsigned int* indices = reinterpret_cast<unsigned int*>(indexBytes.data());
	*indices++ = 0;
	*indices++ = 1;

	Qt3DRender::QBuffer* indexBuffer = new Qt3DRender::QBuffer(geometry);
	indexBuffer->setData(indexBytes);

	Qt3DRender::QAttribute* indexAttribute = new Qt3DRender::QAttribute(geometry);
	indexAttribute->setVertexBaseType(Qt3DRender::QAttribute::UnsignedInt);
	indexAttribute->setAttributeType(Qt3DRender::QAttribute::IndexAttribute);
	indexAttribute->setBuffer(indexBuffer);
	indexAttribute->setCount(2);
	geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

	// mesh
	Qt3DRender::QGeometryRenderer* line = new Qt3DRender::QGeometryRenderer();
	line->setGeometry(geometry);
	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
	Qt3DExtras::QPhongMaterial* material = new Qt3DExtras::QPhongMaterial();
	material->setAmbient(color);

	// entity
	Qt3DCore::QEntity* lineEntity = new Qt3DCore::QEntity(parentEntity);
	lineEntity->addComponent(line);
	lineEntity->addComponent(material);

	if (showArrow)
	{
		createCone(dir, end, color, lineEntity);
	}
	return lineEntity;
}

Qt3DCore::QEntity* LineViewer::createLine(const QVector3D& start, const QVector3D& end, const QColor& color, qreal width, Qt3DCore::QEntity* _rootEntity)
{
	QVector3D diff = end - start;
	qreal length = diff.length();
	QVector3D pos = (start + end) / 2;
	return nullptr;
}

void LineViewer::resetCamera()
{
	// Camera
	Qt3DRender::QCamera* camera = m_canvas->camera();
	camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
	camera->setPosition(QVector3D(0, 0, 5.0f));
	camera->setViewCenter(QVector3D(0, 0, 0));

	m_camController = new Qt3DExtras::QOrbitCameraController(m_rootEntity);
	m_camController->setLinearSpeed(20.0f);
	m_camController->setLookSpeed(90.0f);
	m_camController->setCamera(camera);
}

void LineViewer::setEntityColor(Qt3DCore::QEntity* entity, QColor color)
{
	if (entity == nullptr)
		return;

	Qt3DExtras::QPhongMaterial* mat = getComponent<Qt3DExtras::QPhongMaterial>(entity);
	if (mat == nullptr)
		return;

	mat->setAmbient(color);
}

void LineViewer::randomGenerate()
{
	clearLines();
	int lines = m_ui->spinBoxLines->value();
	QVector3D min(m_ui->spinBoxRangeMinX->value(), m_ui->spinBoxRangeMinY->value(), m_ui->spinBoxRangeMinZ->value());
	QVector3D max(m_ui->spinBoxRangeMaxX->value(), m_ui->spinBoxRangeMaxY->value(), m_ui->spinBoxRangeMaxZ->value());

	QRandomGenerator* random = QRandomGenerator::global();
	float rangeX = max.x() - min.x();
	float rangeY = max.y() - min.y();
	float rangeZ = max.y() - min.z();
	for (int i = 0; i < lines; i++)
	{
		float startX = random->bounded(rangeX) + min.x();
		float startY = random->bounded(rangeY) + min.y();
		float startZ = random->bounded(rangeZ) + min.z();
		float endX = random->bounded(rangeX) + min.x();
		float endY = random->bounded(rangeY) + min.y();
		float endZ = random->bounded(rangeZ) + min.z();

		QVector3D start(startX, startY, startZ);
		QVector3D end(endX, endY, endZ);
		QVector3D v = end - start;
		float length = v.length();
		qDebug() << "line" << i << ":" << start << end << length;

		Qt3DCore::QEntity* line = createLine(start, end, Qt::blue, m_rootEntity, true);
		m_lines.append(line);
		LineSegment ls(Eigen::Vector3f(startX, startY, startZ), Eigen::Vector3f(endX, endY, endZ), i);
		ls.generateLocalDescriptor();
		m_linesCloud->points.push_back(ls);
	}

	m_linesTree->setInputCloud(m_linesCloud);
}

void LineViewer::arrangedGenerate()
{
	clearLines();
	int lines = m_ui->spinBoxLines->value();
	QVector3D min(m_ui->spinBoxRangeMinX->value(), m_ui->spinBoxRangeMinY->value(), m_ui->spinBoxRangeMinZ->value());
	QVector3D max(m_ui->spinBoxRangeMaxX->value(), m_ui->spinBoxRangeMaxY->value(), m_ui->spinBoxRangeMaxZ->value());

	QRandomGenerator* random = QRandomGenerator::global();
	float rangeX = max.x() - min.x();
	float rangeY = max.y() - min.y();
	float rangeZ = max.y() - min.z();
	QVector3D offset(0.0f, -0.1f, 0.0f);
	QVector3D delta(0.1f, 0.0f, 0.0f);
	QVector3D begin(-0.5f, 0.0f, 0.0f);
	QVector3D dir(0.0f, 0.0f, 1.0f);
	float length = 1.0f;
	for (int i = 0; i < lines; i++)
	{
		QVector3D start = begin + delta * i + offset;
		QVector3D end = start + dir * length;
		if (i % 2 == 0)
		{
			qSwap<QVector3D>(start, end);
		}

		Qt3DCore::QEntity* line = createLine(start, end, Qt::blue, m_rootEntity, true);
		m_lines.append(line);
		LineSegment ls1(Eigen::Vector3f(start.x(), start.y(), start.z()), Eigen::Vector3f(end.x(), end.y(), end.z()), i);
		LineSegment ls2(Eigen::Vector3f(end.x(), end.y(), end.z()), Eigen::Vector3f(start.x(), start.y(), start.z()), i);
		ls1.generateLocalDescriptor();
		ls1.showLocalDescriptor();
		ls2.generateLocalDescriptor();
		ls2.showLocalDescriptor();
		m_linesCloud->points.push_back(ls1);
		m_linesCloud->points.push_back(ls2);
	}
	m_linesTree->setInputCloud(m_linesCloud);
}

void LineViewer::clearLines()
{
	m_linesCloud->clear();
	for (Qt3DCore::QEntity* line : m_lines)
	{
		m_rootEntity->childNodes().removeOne(line);
		line->deleteLater();
	}
	m_lines.clear();
}

void LineViewer::searchLine()
{
	if (m_lines.isEmpty())
		return;
	std::vector<int> indices;
	std::vector<float> distances;
	m_linesTree->nearestKSearch(m_line, 1, indices, distances);

	LineSegment line = m_linesCloud->points[indices[0]];
	qDebug() << "found line:" << line.segmentNo() << ", distance:" << distances[0];
	line.showLocalDescriptor();

	Qt3DCore::QEntity* lineEntity = m_lines[line.segmentNo()];
	if (m_foundLine)
		setEntityColor(m_foundLine, Qt::blue);
	setEntityColor(lineEntity, Qt::darkRed);

	m_foundLine = lineEntity;
}

void LineViewer::closeEvent(QCloseEvent* event)
{
	QMainWindow::closeEvent(event);
	this->deleteLater();
	qDebug() << "Destroying Line Viewer...";
}

void LineViewer::onObjectPicked(Qt3DRender::QPickEvent* pick)
{
	qDebug() << pick->entity()->objectName();
	m_currentObject = pick->entity();
}

void LineViewer::updateTransform(Qt3DCore::QTransform* transform, qreal pitch, qreal yaw, qreal roll, qreal x, qreal y, qreal z, bool abstract)
{
	if (transform == nullptr)
		return;
	QQuaternion rot = QQuaternion::fromEulerAngles(pitch, yaw, roll);
	QVector3D pos(x, y, z);
	QMatrix4x4 matrix;
	matrix.setToIdentity();
	matrix.rotate(rot);
	if (abstract)
	{
		matrix.setColumn(3, QVector4D(pos, 1));
	}
	else
	{
		matrix.translate(pos);
	}
	transform->setMatrix(matrix);

	QVector3D dir(0, 0, 1);
	dir = rot * dir;
	QVector3D newPos = matrix * QVector3D(0, 0, 0);
	m_ui->labelLineDir->setText(QString("%1, %2, %3").arg(dir.x()).arg(dir.y()).arg(dir.z()));

	Eigen::Vector3f eDir(dir.x(), dir.y(), dir.z());
	Eigen::Vector3f start(newPos.x(), newPos.y(), newPos.z());
	Eigen::Vector3f end = start + eDir;
	m_line = LineSegment (start, end);
	m_line.generateLocalDescriptor();
	m_line.showLocalDescriptor();

	Eigen::Vector2f angles = m_line.calculateAngles();
	m_ui->labelAngles->setText(QString("%1, %2").arg(qRadiansToDegrees(angles.x())).arg(qRadiansToDegrees(angles.y())));
}

void LineViewer::onTransformValuesChanged(double /*value*/)
{
	if (!m_currentObject)
		return;
	updateTransform(getComponent<Qt3DCore::QTransform>(m_currentObject), 
		-m_ui->doubleSpinBoxPitch->value(), -m_ui->doubleSpinBoxYaw->value(), m_ui->doubleSpinBoxRoll->value(),
		m_ui->doubleSpinBoxTransX->value(), m_ui->doubleSpinBoxTransY->value(), m_ui->doubleSpinBoxTransZ->value(), 
		m_ui->checkBoxAbstract->isChecked());
	searchLine();
}
