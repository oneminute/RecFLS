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

#ifndef IMAGEVIEWER_H_
#define IMAGEVIEWER_H_

#include <QGraphicsView>
#include <QtCore/QRectF>
#include <QtCore/QMultiMap>
#include <QtCore/QSettings>
#include <opencv2/features2d/features2d.hpp>
#include <map>
#include "util/Utils.h"

class QAction;
class QMenu;

class ImageViewer : public QWidget {

	Q_OBJECT

public:
    ImageViewer(QWidget * parent = nullptr);
    virtual ~ImageViewer();

	void saveSettings(QSettings & settings, const QString & group = "") const;
	void loadSettings(QSettings & settings, const QString & group = "");

	QRectF sceneRect() const;
	bool isImageShown() const;
	bool isImageDepthShown() const;
	bool isFeaturesShown() const;
	bool isLinesShown() const;
	int getAlpha() const {return _alpha;}
	int getFeaturesSize() const {return _featuresSize;}
	bool isGraphicsViewMode() const;
	bool isGraphicsViewScaled() const;
	bool isGraphicsViewScaledToHeight() const;
	const QColor & getDefaultBackgroundColor() const;
	const QColor & getDefaultFeatureColor() const;
	const QColor & getDefaultMatchingFeatureColor() const;
	const QColor & getDefaultMatchingLineColor() const;
	const QColor & getBackgroundColor() const;
    uCvQtDepthColorMap getDepthColorMap() const;

	float viewScale() const;

	void setImageShown(bool shown);
	void setImageDepthShown(bool shown);
	void setLinesShown(bool shown);
	void setGraphicsViewMode(bool on);
	void setGraphicsViewScaled(bool scaled);
	void setGraphicsViewScaledToHeight(bool scaled);
	void setDefaultBackgroundColor(const QColor & color);
	void setDefaultFeatureColor(const QColor & color);
	void setDefaultMatchingFeatureColor(const QColor & color);
	void setDefaultMatchingLineColor(const QColor & color);
	void setBackgroundColor(const QColor & color);

	void addLine(float x1, float y1, float x2, float y2, QColor color, const QString & text = QString());
	void setImage(const QImage & image);
	void setImageDepth(const cv::Mat & imageDepth);
	void setImageDepth(const QImage & image);
	void setSceneRect(const QRectF & rect);

	void clearLines();
	void clear();

	virtual QSize sizeHint() const;

Q_SIGNALS:
	void configChanged();

protected:
	virtual void paintEvent(QPaintEvent *event);
	virtual void resizeEvent(QResizeEvent* event);
	virtual void contextMenuEvent(QContextMenuEvent * e);

private Q_SLOTS:
	void sceneRectChanged(const QRectF &rect);

private:
	void updateOpacity();
	void computeScaleOffsets(const QRect & targetRect, float & scale, float & offsetX, float & offsetY) const;
	QIcon createIcon(const QColor & color);

private:
	QString _savedFileName;
	int _alpha;
	int _featuresSize;
	QColor _defaultBgColor;
	QColor _defaultFeatureColor;
	QColor _defaultMatchingFeatureColor;
	QColor _defaultMatchingLineColor;

	QMenu * _menu;
	QAction * _showImage;
	QAction * _showImageDepth;
	QAction * _showFeatures;
	QAction * _showLines;
	QAction * _setFeatureColor;
	QAction * _setMatchingFeatureColor;
	QAction * _setMatchingLineColor;
	QAction * _saveImage;
	QAction * _setAlpha;
	QAction * _setFeaturesSize;
	QAction * _graphicsViewMode;
	QAction * _graphicsViewScaled;
	QAction * _graphicsViewScaledToHeight;
	QAction * _graphicsViewNoScaling;
	QAction * _colorMapWhiteToBlack;
	QAction * _colorMapBlackToWhite;
	QAction * _colorMapRedToBlue;
	QAction * _colorMapBlueToRed;
	QMenu * _featureMenu;
	QMenu * _scaleMenu;

	QGraphicsView * _graphicsView;
	QList<QGraphicsLineItem*> _lines;
	QGraphicsPixmapItem * _imageItem;
	QGraphicsPixmapItem * _imageDepthItem;
	QPixmap _image;
	QPixmap _imageDepth;
	cv::Mat _imageDepthCv;
};


#endif /* IMAGEVIEWER_H_ */
