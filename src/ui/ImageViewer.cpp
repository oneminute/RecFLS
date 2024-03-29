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

#include "ImageViewer.h"

#include <QtGui/QWheelEvent>
#include <QtCore/qmath.h>
#include <QMenu>
#include <QFileDialog>
#include <QtCore/QDir>
#include <QAction>
#include <QGraphicsEffect>
#include <QInputDialog>
#include <QVBoxLayout>
#include <QColorDialog>
#include <QGraphicsRectItem>

//LineItem
class LineItem : public QGraphicsLineItem
{
public:
    LineItem(float x1, float y1, float x2, float y2, const QString & text = QString(), QGraphicsItem * parent = nullptr) :
		QGraphicsLineItem(x1, y1, x2, y2, parent),
		_text(text),
        _placeHolder(nullptr)
	{
		this->setAcceptHoverEvents(true);
		this->setFlag(QGraphicsItem::ItemIsFocusable, true);
		_width = pen().width();
	}
	virtual ~LineItem()
	{
		delete _placeHolder;
	}

	void setColor(const QColor & color);

protected:
	virtual void hoverEnterEvent ( QGraphicsSceneHoverEvent * event )
	{
		QGraphicsScene * scene = this->scene();
        if(scene && scene->focusItem() == nullptr)
		{
			this->showDescription();
		}
		else
		{
			this->setPen(QPen(pen().color(), _width+2));
		}
		QGraphicsLineItem::hoverEnterEvent(event);
	}

	virtual void hoverLeaveEvent ( QGraphicsSceneHoverEvent * event )
	{
		if(!this->hasFocus())
		{
			this->hideDescription();
		}
		QGraphicsLineItem::hoverEnterEvent(event);
	}

	virtual void focusInEvent ( QFocusEvent * event )
	{
		this->showDescription();
		QGraphicsLineItem::focusInEvent(event);
	}

	virtual void focusOutEvent ( QFocusEvent * event )
	{
		this->hideDescription();
		QGraphicsLineItem::focusOutEvent(event);
	}

private:
	void showDescription()
	{
		if(!_text.isEmpty())
		{
			if(!_placeHolder)
			{
				_placeHolder = new QGraphicsRectItem (this);
				_placeHolder->setVisible(false);
				if(qGray(pen().color().rgb() > 255/2))
				{
					_placeHolder->setBrush(QBrush(QColor ( 0,0,0, 170 )));
				}
				else
				{
					_placeHolder->setBrush(QBrush(QColor ( 255, 255, 255, 170 )));
				}
				QGraphicsTextItem * text = new QGraphicsTextItem(_placeHolder);
				text->setDefaultTextColor(this->pen().color().rgb());
				text->setPlainText(_text);
				_placeHolder->setRect(text->boundingRect());
			}

			if(_placeHolder->parentItem())
			{
				_placeHolder->setParentItem(0); // Make it a to level item
			}
			_placeHolder->setZValue(this->zValue()+1);
			_placeHolder->setPos(this->mapFromScene(0,0));
			_placeHolder->setVisible(true);
		}
		QPen pen = this->pen();
		this->setPen(QPen(pen.color(), _width+2));
	}
	void hideDescription()
	{
		if(_placeHolder)
		{
			_placeHolder->setVisible(false);
		}
		this->setPen(QPen(pen().color(), _width));
	}

private:
	QString _text;
	QGraphicsRectItem * _placeHolder;
	int _width;
};

QIcon ImageViewer::createIcon(const QColor & color)
{
	QPixmap pixmap(50, 50);
	pixmap.fill(color);
	return QIcon(pixmap);
}

ImageViewer::ImageViewer(QWidget * parent) :
		QWidget(parent),
		_savedFileName((QDir::homePath()+ "/") + "picture" + ".png"),
		_alpha(50),
		_featuresSize(0.0f),
		_defaultBgColor(Qt::black),
		_defaultFeatureColor(Qt::yellow),
		_defaultMatchingFeatureColor(Qt::magenta),
		_defaultMatchingLineColor(Qt::cyan),
        _imageItem(nullptr),
        _imageDepthItem(nullptr)
{
	_graphicsView = new QGraphicsView(this);
	_graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	_graphicsView->setScene(new QGraphicsScene(this));
	_graphicsView->setVisible(false);

	this->setLayout(new QVBoxLayout(this));
	this->layout()->addWidget(_graphicsView);
	this->layout()->setContentsMargins(0,0,0,0);

	_menu = new QMenu(tr(""), this);
	_showImage = _menu->addAction(tr("Show image"));
	_showImage->setCheckable(true);
	_showImage->setChecked(true);
	_showImageDepth = _menu->addAction(tr("Show image depth"));
	_showImageDepth->setCheckable(true);
	_showImageDepth->setChecked(false);
	_featureMenu = _menu->addMenu("Features");
	_showFeatures = _featureMenu->addAction(tr("Show features"));
	_showFeatures->setCheckable(true);
	_showFeatures->setChecked(true);
	_setFeaturesSize = _featureMenu->addAction(tr("Set features size..."));
	_showLines = _featureMenu->addAction(tr("Show lines"));
	_showLines->setCheckable(true);
	_showLines->setChecked(true);
	_setFeatureColor = _featureMenu->addAction(tr("Set default feature color"));
	_setFeatureColor->setIcon(createIcon(_defaultFeatureColor));
	_setFeatureColor->setIconVisibleInMenu(true);
	_setMatchingFeatureColor = _featureMenu->addAction(tr("Set default correspondence color"));
	_setMatchingFeatureColor->setIcon(createIcon(_defaultMatchingFeatureColor));
	_setMatchingFeatureColor->setIconVisibleInMenu(true);
	_setMatchingLineColor = _featureMenu->addAction(tr("Set default line color"));
	_setMatchingLineColor->setIcon(createIcon(_defaultMatchingLineColor));
	_setMatchingLineColor->setIconVisibleInMenu(true);
	_setAlpha = _featureMenu->addAction(tr("Set transparency..."));
	_graphicsViewMode = _menu->addAction(tr("Graphics view"));
	_graphicsViewMode->setCheckable(true);
	_graphicsViewMode->setChecked(false);
	_scaleMenu = _menu->addMenu("Scale image");
	_scaleMenu->setEnabled(false);
	_graphicsViewScaled = _scaleMenu->addAction(tr("Fit in view"));
	_graphicsViewScaled->setCheckable(true);
	_graphicsViewScaled->setChecked(true);
	_graphicsViewScaledToHeight = _scaleMenu->addAction(tr("Fit height"));
	_graphicsViewScaledToHeight->setCheckable(true);
	_graphicsViewScaledToHeight->setChecked(false);
	_graphicsViewNoScaling = _scaleMenu->addAction(tr("No scale"));
	_graphicsViewNoScaling->setCheckable(true);
	_graphicsViewNoScaling->setChecked(false);
	QActionGroup * group = new QActionGroup(this);
	group->addAction(_graphicsViewScaled);
	group->addAction(_graphicsViewScaledToHeight);
	group->addAction(_graphicsViewNoScaling);
	QMenu * colorMap = _menu->addMenu("Depth color map");
	_colorMapWhiteToBlack = colorMap->addAction(tr("White to black"));
	_colorMapWhiteToBlack->setCheckable(true);
	_colorMapWhiteToBlack->setChecked(false);
	_colorMapBlackToWhite = colorMap->addAction(tr("Black to white"));
	_colorMapBlackToWhite->setCheckable(true);
	_colorMapBlackToWhite->setChecked(false);
	_colorMapRedToBlue = colorMap->addAction(tr("Red to blue"));
	_colorMapRedToBlue->setCheckable(true);
	_colorMapRedToBlue->setChecked(true);
	_colorMapBlueToRed = colorMap->addAction(tr("Blue to red"));
	_colorMapBlueToRed->setCheckable(true);
	_colorMapBlueToRed->setChecked(false);
	group = new QActionGroup(this);
	group->addAction(_colorMapWhiteToBlack);
	group->addAction(_colorMapBlackToWhite);
	group->addAction(_colorMapRedToBlue);
	group->addAction(_colorMapBlueToRed);
	_saveImage = _menu->addAction(tr("Save picture..."));
	_saveImage->setEnabled(false);

	connect(_graphicsView->scene(), SIGNAL(sceneRectChanged(const QRectF &)), this, SLOT(sceneRectChanged(const QRectF &)));
}

ImageViewer::~ImageViewer() {
	clear();
}

void ImageViewer::saveSettings(QSettings & settings, const QString & group) const
{
	if(!group.isEmpty())
	{
		settings.beginGroup(group);
	}
	settings.setValue("image_shown", this->isImageShown());
	settings.setValue("depth_shown", this->isImageDepthShown());
	settings.setValue("features_shown", this->isFeaturesShown());
	settings.setValue("features_size", this->getFeaturesSize());
	settings.setValue("lines_shown", this->isLinesShown());
	settings.setValue("alpha", this->getAlpha());
	settings.setValue("bg_color", this->getDefaultBackgroundColor());
	settings.setValue("feature_color", this->getDefaultFeatureColor());
	settings.setValue("matching_feature_color", this->getDefaultMatchingFeatureColor());
	settings.setValue("matching_line_color", this->getDefaultMatchingLineColor());
	settings.setValue("graphics_view", this->isGraphicsViewMode());
	settings.setValue("graphics_view_scale", this->isGraphicsViewScaled());
	settings.setValue("graphics_view_scale_to_height", this->isGraphicsViewScaledToHeight());
	settings.setValue("colormap", _colorMapWhiteToBlack->isChecked()?0:_colorMapBlackToWhite->isChecked()?1:_colorMapRedToBlue->isChecked()?2:3);
	if(!group.isEmpty())
	{
		settings.endGroup();
	}
}

void ImageViewer::loadSettings(QSettings & settings, const QString & group)
{
	if(!group.isEmpty())
	{
		settings.beginGroup(group);
	}
	this->setImageShown(settings.value("image_shown", this->isImageShown()).toBool());
	this->setImageDepthShown(settings.value("depth_shown", this->isImageDepthShown()).toBool());
	this->setLinesShown(settings.value("lines_shown", this->isLinesShown()).toBool());
	this->setDefaultBackgroundColor(settings.value("bg_color", this->getDefaultBackgroundColor()).value<QColor>());
	this->setDefaultFeatureColor(settings.value("feature_color", this->getDefaultFeatureColor()).value<QColor>());
	this->setDefaultMatchingFeatureColor(settings.value("matching_feature_color", this->getDefaultMatchingFeatureColor()).value<QColor>());
	this->setDefaultMatchingLineColor(settings.value("matching_line_color", this->getDefaultMatchingLineColor()).value<QColor>());
	this->setGraphicsViewMode(settings.value("graphics_view", this->isGraphicsViewMode()).toBool());
	this->setGraphicsViewScaled(settings.value("graphics_view_scale", this->isGraphicsViewScaled()).toBool());
	this->setGraphicsViewScaledToHeight(settings.value("graphics_view_scale_to_height", this->isGraphicsViewScaledToHeight()).toBool());
	int colorMap = settings.value("colormap", 0).toInt();
	_colorMapWhiteToBlack->setChecked(colorMap==0);
	_colorMapBlackToWhite->setChecked(colorMap==1);
	_colorMapRedToBlue->setChecked(colorMap==2);
	_colorMapBlueToRed->setChecked(colorMap==3);
	if(!group.isEmpty())
	{
		settings.endGroup();
	}
}

QRectF ImageViewer::sceneRect() const
{
	return _graphicsView->scene()->sceneRect();
}

bool ImageViewer::isImageShown() const
{
	return _showImage->isChecked();
}

bool ImageViewer::isImageDepthShown() const
{
	return _showImageDepth->isChecked();
}

bool ImageViewer::isFeaturesShown() const
{
	return _showFeatures->isChecked();
}

bool ImageViewer::isGraphicsViewMode() const
{
	return _graphicsViewMode->isChecked();
}

bool ImageViewer::isGraphicsViewScaled() const
{
	return _graphicsViewScaled->isChecked();
}

bool ImageViewer::isGraphicsViewScaledToHeight() const
{
	return _graphicsViewScaledToHeight->isChecked();
}

const QColor & ImageViewer::getDefaultBackgroundColor() const
{
	return _defaultBgColor;
}
const QColor & ImageViewer::getDefaultFeatureColor() const
{
	return _defaultFeatureColor;
}
const QColor & ImageViewer::getDefaultMatchingFeatureColor() const
{
	return _defaultMatchingFeatureColor;
}
const QColor & ImageViewer::getDefaultMatchingLineColor() const
{
	return _defaultMatchingLineColor;
}

const QColor & ImageViewer::getBackgroundColor() const
{
    return _graphicsView->backgroundBrush().color();
}

uCvQtDepthColorMap ImageViewer::getDepthColorMap() const
{
    uCvQtDepthColorMap colorMap = uCvQtDepthWhiteToBlack;
    if(_colorMapBlackToWhite->isChecked())
    {
        colorMap = uCvQtDepthBlackToWhite;
    }
    else if(_colorMapRedToBlue->isChecked())
    {
        colorMap = uCvQtDepthRedToBlue;
    }
    else if(_colorMapBlueToRed->isChecked())
    {
        colorMap = uCvQtDepthBlueToRed;
    }
    return colorMap;
}


void ImageViewer::setImageShown(bool shown)
{
	_showImage->setChecked(shown);
	if(_imageItem)
	{
		_imageItem->setVisible(_showImage->isChecked());
		this->updateOpacity();
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::setImageDepthShown(bool shown)
{
	_showImageDepth->setChecked(shown);
	if(_imageDepthItem)
	{
		_imageDepthItem->setVisible(_showImageDepth->isChecked());
		this->updateOpacity();
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

bool ImageViewer::isLinesShown() const
{
	return _showLines->isChecked();
}

void ImageViewer::setLinesShown(bool shown)
{
	_showLines->setChecked(shown);
	for(int i=0; i<_lines.size(); ++i)
	{
		_lines.at(i)->setVisible(_showLines->isChecked());
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

float ImageViewer::viewScale() const
{
	if(_graphicsView->isVisible())
	{
		return _graphicsView->transform().m11();
	}
	else
	{
		float scale, offsetX, offsetY;
		computeScaleOffsets(this->rect(), scale, offsetX, offsetY);
		return scale;
	}
}

void ImageViewer::setGraphicsViewMode(bool on)
{
	_graphicsViewMode->setChecked(on);
	_graphicsView->setVisible(on);
	_scaleMenu->setEnabled(on);

	if(on)
	{
		for(QList<QGraphicsLineItem*>::iterator iter=_lines.begin(); iter!=_lines.end(); ++iter)
		{
			_graphicsView->scene()->addItem(*iter);
		}

		//update images
		if(_imageItem)
		{
			_imageItem->setPixmap(_image);
		}
		else
		{
			_imageItem = _graphicsView->scene()->addPixmap(_image);
			_imageItem->setVisible(_showImage->isChecked());
		}

		if(_imageDepthItem)
		{
			_imageDepthItem->setPixmap(_imageDepth);
		}
		else
		{
			_imageDepthItem = _graphicsView->scene()->addPixmap(_imageDepth);
			_imageDepthItem->setVisible(_showImageDepth->isChecked());
		}
		this->updateOpacity();

		if(_graphicsViewScaled->isChecked())
		{
			_graphicsView->fitInView(_graphicsView->sceneRect(), Qt::KeepAspectRatio);
		}
		else if(_graphicsViewScaledToHeight->isChecked())
		{
			QRectF rect = _graphicsView->sceneRect();
			rect.setWidth(1);
			_graphicsView->fitInView(rect, Qt::KeepAspectRatio);
		}
		else
		{
			_graphicsView->resetTransform();
		}
	}
	else
	{
		this->update();
	}
}

void ImageViewer::setGraphicsViewScaled(bool scaled)
{
	_graphicsViewScaled->setChecked(scaled);

	if(scaled)
	{
		_graphicsView->fitInView(_graphicsView->sceneRect(), Qt::KeepAspectRatio);
	}
	else
	{
		_graphicsView->resetTransform();
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::setGraphicsViewScaledToHeight(bool scaled)
{
	_graphicsViewScaledToHeight->setChecked(scaled);

	if(scaled)
	{
		QRectF rect = _graphicsView->sceneRect();
		rect.setWidth(1);
		_graphicsView->fitInView(rect, Qt::KeepAspectRatio);
	}
	else
	{
		_graphicsView->resetTransform();
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::setDefaultBackgroundColor(const QColor & color)
{
	_defaultBgColor = color;
	setBackgroundColor(color);
}

void ImageViewer::setDefaultFeatureColor(const QColor & color)
{
	_defaultFeatureColor = color;
}
void ImageViewer::setDefaultMatchingFeatureColor(const QColor & color)
{
	_defaultMatchingFeatureColor = color;
}
void ImageViewer::setDefaultMatchingLineColor(const QColor & color)
{
	_defaultMatchingLineColor = color;
}

void ImageViewer::setBackgroundColor(const QColor & color)
{
	_graphicsView->setBackgroundBrush(QBrush(color));

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::computeScaleOffsets(const QRect & targetRect, float & scale, float & offsetX, float & offsetY) const
{
	scale = 1.0f;
	offsetX = 0.0f;
	offsetY = 0.0f;

	if(!_graphicsView->scene()->sceneRect().isNull())
	{
		float w = _graphicsView->scene()->width();
		float h = _graphicsView->scene()->height();
		float widthRatio = float(targetRect.width()) / w;
		float heightRatio = float(targetRect.height()) / h;

		//printf("w=%f, h=%f, wR=%f, hR=%f, sW=%d, sH=%d\n", w, h, widthRatio, heightRatio, this->rect().width(), this->rect().height());
		if(widthRatio < heightRatio)
		{
			scale = widthRatio;
		}
		else
		{
			scale = heightRatio;
		}

		//printf("ratio=%f\n",ratio);

		w *= scale;
		h *= scale;

		if(w < targetRect.width())
		{
			offsetX = (targetRect.width() - w)/2.0f;
		}
		if(h < targetRect.height())
		{
			offsetY = (targetRect.height() - h)/2.0f;
		}
		//printf("offsetX=%f, offsetY=%f\n",offsetX, offsetY);
	}
}

void ImageViewer::sceneRectChanged(const QRectF & rect)
{
	_saveImage->setEnabled(rect.isValid());
}

void ImageViewer::paintEvent(QPaintEvent *event)
{
	if(_graphicsViewMode->isChecked())
	{
		QWidget::paintEvent(event);
	}
	else
	{
		if(!_graphicsView->scene()->sceneRect().isNull())
		{
			//Scale
			float ratio, offsetX, offsetY;
			this->computeScaleOffsets(event->rect(), ratio, offsetX, offsetY);
			QPainter painter(this);

			//Background
			painter.save();
			painter.setBrush(_graphicsView->backgroundBrush());
			painter.drawRect(event->rect());
			painter.restore();

			painter.translate(offsetX, offsetY);
			painter.scale(ratio, ratio);

			painter.save();
			if(_showImage->isChecked() && !_image.isNull() &&
			   _showImageDepth->isChecked() && !_imageDepth.isNull())
			{
				painter.setOpacity(0.5);
			}

			if(_showImage->isChecked() && !_image.isNull())
			{
				painter.drawPixmap(QPoint(0,0), _image);
			}

			if(_showImageDepth->isChecked() && !_imageDepth.isNull())
			{
				painter.drawPixmap(QPoint(0,0), _imageDepth);
			}
			painter.restore();

			if(_showLines->isChecked())
			{
				for(QList<QGraphicsLineItem*>::iterator iter = _lines.begin(); iter != _lines.end(); ++iter)
				{
					QColor color = (*iter)->pen().color();
					painter.save();
					painter.setPen(color);
					painter.drawLine((*iter)->line());
					painter.restore();
				}
			}
		}
	}
}

void ImageViewer::resizeEvent(QResizeEvent* event)
{
	QWidget::resizeEvent(event);
	if(_graphicsView->isVisible())
	{
		if(_graphicsViewScaled->isChecked())
		{
			_graphicsView->fitInView(_graphicsView->sceneRect(), Qt::KeepAspectRatio);
		}
		else if(_graphicsViewScaledToHeight->isChecked())
		{
			QRectF rect = _graphicsView->sceneRect();
			rect.setWidth(1);
			_graphicsView->fitInView(rect, Qt::KeepAspectRatio);
		}
	}
}

void ImageViewer::contextMenuEvent(QContextMenuEvent * e)
{
	_setFeatureColor->setIcon(createIcon(_defaultFeatureColor));
	_setMatchingFeatureColor->setIcon(createIcon(_defaultMatchingFeatureColor));
	_setMatchingLineColor->setIcon(createIcon(_defaultMatchingLineColor));
	_setFeatureColor->setIconVisibleInMenu(true);
	_setMatchingFeatureColor->setIconVisibleInMenu(true);
	_setMatchingLineColor->setIconVisibleInMenu(true);

	QAction * action = _menu->exec(e->globalPos());
	if(action == _saveImage)
	{
		if(!_graphicsView->scene()->sceneRect().isNull())
		{
			QString text;
#ifdef QT_SVG_LIB
			text = QFileDialog::getSaveFileName(this, tr("Save figure to ..."), _savedFileName, "*.png *.xpm *.jpg *.pdf *.svg");
#else
			text = QFileDialog::getSaveFileName(this, tr("Save figure to ..."), _savedFileName, "*.png *.xpm *.jpg *.pdf");
#endif
			if(!text.isEmpty())
			{
				if(QFileInfo(text).suffix() == "")
				{
					//use png by default
					text += ".png";
				}

				_savedFileName = text;
				QImage img(_graphicsView->sceneRect().width(), _graphicsView->sceneRect().height(), QImage::Format_ARGB32_Premultiplied);
				QPainter p(&img);
				if(_graphicsView->isVisible())
				{
					_graphicsView->scene()->render(&p, _graphicsView->sceneRect(), _graphicsView->sceneRect());
				}
				else
				{
					this->render(&p, QPoint(), _graphicsView->sceneRect().toRect());
				}
				img.save(text);
			}
		}
	}
	else if(action == _setFeatureColor ||
			action == _setMatchingFeatureColor ||
			action == _setMatchingLineColor)
	{
		QColor color;
		if(action == _setMatchingFeatureColor)
		{
			color = _defaultMatchingFeatureColor;
		}
		else if(action == _setMatchingLineColor)
		{
			color = _defaultMatchingLineColor;
		}
		else //if(action == _setFeatureColor)
		{
			color = _defaultFeatureColor;
		}
		color = QColorDialog::getColor(color, this);
		if(color.isValid())
		{

			if(action == _setMatchingFeatureColor)
			{
				this->setDefaultMatchingFeatureColor(color);
			}
			else if(action == _setMatchingLineColor)
			{
				this->setDefaultMatchingLineColor(color);
			}
			else //if(action == _setFeatureColor)
			{
				this->setDefaultFeatureColor(color);
			}
		}
	}
	else if(action == _showImage)
	{
		this->setImageShown(_showImage->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _showImageDepth)
	{
		this->setImageDepthShown(_showImageDepth->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _showLines)
	{
		this->setLinesShown(_showLines->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _graphicsViewMode)
	{
		this->setGraphicsViewMode(_graphicsViewMode->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _graphicsViewScaled)
	{
		this->setGraphicsViewScaled(_graphicsViewScaled->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _graphicsViewScaledToHeight || action == _graphicsViewNoScaling)
	{
		this->setGraphicsViewScaledToHeight(_graphicsViewScaledToHeight->isChecked());
		Q_EMIT configChanged();
	}
	else if(action == _colorMapBlackToWhite || action == _colorMapWhiteToBlack || action == _colorMapRedToBlue || action == _colorMapBlueToRed)
	{
		if(!_imageDepthCv.empty())
			this->setImageDepth(_imageDepthCv);
		Q_EMIT configChanged();
	}

	if(action == _showImage || action ==_showImageDepth)
	{
		this->updateOpacity();
		Q_EMIT configChanged();
	}
}

void ImageViewer::updateOpacity()
{
	if(_imageItem && _imageDepthItem)
	{
		if(_imageItem->isVisible() && _imageDepthItem->isVisible())
		{
			QGraphicsOpacityEffect * effect = new QGraphicsOpacityEffect();
			effect->setOpacity(0.5);
			_imageDepthItem->setGraphicsEffect(effect);
		}
		else
		{
			_imageDepthItem->setGraphicsEffect(0);
		}
	}
	else if(_imageDepthItem)
	{
		_imageDepthItem->setGraphicsEffect(0);
	}
}

void ImageViewer::addLine(float x1, float y1, float x2, float y2, QColor color, const QString & text)
{
	color.setAlpha(this->getAlpha());
	LineItem * item  = new LineItem(x1, y1, x2, y2, text);
	item->setPen(QPen(color));
	_lines.push_back(item);
	item->setVisible(isLinesShown());
	item->setZValue(1);

	if(_graphicsView->isVisible())
	{
		_graphicsView->scene()->addItem(item);
	}
}

void ImageViewer::setImage(const QImage & image)
{
	_image = QPixmap::fromImage(image);
	if(_graphicsView->isVisible())
	{
		if(_imageItem)
		{
			_imageItem->setPixmap(_image);
		}
		else
		{
			_imageItem = _graphicsView->scene()->addPixmap(_image);
			_imageItem->setVisible(_showImage->isChecked());
			this->updateOpacity();
		}
	}

	if(image.rect().isValid())
	{
		this->setSceneRect(image.rect());
	}
	else if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::setImageDepth(const cv::Mat & imageDepth)
{
	_imageDepthCv = imageDepth;
    setImageDepth(cvMat2QImage(_imageDepthCv, true, getDepthColorMap()));
}

void ImageViewer::setImageDepth(const QImage & imageDepth)
{
	_imageDepth = QPixmap::fromImage(imageDepth);

    Q_ASSERT(_imageDepth.width() && _imageDepth.height());

	if( _image.width() > 0 &&
		_image.width() > _imageDepth.width() &&
		_image.height() > _imageDepth.height() &&
		_image.width() % _imageDepth.width() == 0 &&
		_image.height() % _imageDepth.height() == 0)
	{
		// scale depth to rgb
		_imageDepth = _imageDepth.scaled(_image.size());
	}

	if(_graphicsView->isVisible())
	{
		if(_imageDepthItem)
		{
			_imageDepthItem->setPixmap(_imageDepth);
		}
		else
		{
			_imageDepthItem = _graphicsView->scene()->addPixmap(_imageDepth);
			_imageDepthItem->setVisible(_showImageDepth->isChecked());
			this->updateOpacity();
		}
	}
	else
	{
		if(_image.isNull())
		{
			this->setSceneRect(imageDepth.rect());
		}
		this->update();
	}
}

void ImageViewer::setSceneRect(const QRectF & rect)
{
	_graphicsView->scene()->setSceneRect(rect);

	if(_graphicsViewScaled->isChecked())
	{
		_graphicsView->fitInView(_graphicsView->sceneRect(), Qt::KeepAspectRatio);
	}
	else if(_graphicsViewScaledToHeight->isChecked())
	{
		QRectF rect = _graphicsView->sceneRect();
		rect.setWidth(1);
		_graphicsView->fitInView(rect, Qt::KeepAspectRatio);
	}
	else
	{
		_graphicsView->resetTransform();
	}

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::clearLines()
{
	qDeleteAll(_lines);
	_lines.clear();

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

void ImageViewer::clear()
{
	qDeleteAll(_lines);
	_lines.clear();

	if(_imageItem)
	{
		_graphicsView->scene()->removeItem(_imageItem);
		delete _imageItem;
		_imageItem = 0;
	}
	_image = QPixmap();

	if(_imageDepthItem)
	{
		_graphicsView->scene()->removeItem(_imageDepthItem);
		delete _imageDepthItem;
		_imageDepthItem = 0;
	}
	_imageDepth = QPixmap();

	_graphicsView->scene()->setSceneRect(QRectF());
	_graphicsView->setScene(_graphicsView->scene());

	if(!_graphicsView->isVisible())
	{
		this->update();
	}
}

QSize ImageViewer::sizeHint() const
{
	return _graphicsView->sizeHint();
}
