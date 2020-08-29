#ifndef PREFERENCSWINDOW_H
#define PREFERENCSWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>
#include <QMap>
#include "common/Parameters.h"

namespace Ui {
class PreferencesWindow;
}

class PreferencesWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit PreferencesWindow(QWidget *parent = nullptr);
    ~PreferencesWindow();

private slots:
    void init();
    void onActionSave(bool checked = false);

private:
    QScopedPointer<Ui::PreferencesWindow> m_ui;

    QMap<QString, QList<SettingItem*>> m_settings;
    
    bool m_init;
};

#endif // PREFERENCSWINDOW_H
