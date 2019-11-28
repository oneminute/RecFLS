#include "recfls.h"
#include "ui_recfls.h"

RecFLS::RecFLS(QWidget *parent) :
    QMainWindow(parent),
    m_ui(new Ui::RecFLS)
{
    m_ui->setupUi(this);
}

RecFLS::~RecFLS() = default;
