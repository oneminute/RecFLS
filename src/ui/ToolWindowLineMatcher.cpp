#include "ToolWindowLineMatcher.h"
#include "ui_ToolWindowLineMatcher.h"

ToolWindowLineMatcher::ToolWindowLineMatcher(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineMatcher)
{
    m_ui->setupUi(this);
}

ToolWindowLineMatcher::~ToolWindowLineMatcher()
{
}
