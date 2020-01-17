#ifndef TOOLWINDOWLINEMATCHER_H
#define TOOLWINDOWLINEMATCHER_H

#include <QMainWindow>
#include <QScopedPointer>

#include "extractor/DDBPLineExtractor.h"
#include "matcher/DDBPLineMatcher.h"

namespace Ui {
class ToolWindowLineMatcher;
}

class ToolWindowLineMatcher : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowLineMatcher(QWidget *parent = nullptr);
    ~ToolWindowLineMatcher();

private:
    QScopedPointer<Ui::ToolWindowLineMatcher> m_ui;
};

#endif // TOOLWINDOWLINEMATCHER_H
