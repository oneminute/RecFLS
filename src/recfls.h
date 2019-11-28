#ifndef RECFLS_H
#define RECFLS_H

#include <QMainWindow>
#include <QScopedPointer>

namespace Ui {
class RecFLS;
}

class RecFLS : public QMainWindow
{
    Q_OBJECT

public:
    explicit RecFLS(QWidget *parent = nullptr);
    ~RecFLS() override;

private:
    QScopedPointer<Ui::RecFLS> m_ui;
};

#endif // RECFLS_H
