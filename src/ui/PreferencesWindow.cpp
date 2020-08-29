#include "PreferencesWindow.h"
#include "ui_PreferencesWindow.h"

#include <QDebug>
#include <QAction>
#include <QFileDialog>
#include <QDir>
#include <QtMath>
#include <QGridLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QSpacerItem>
#include <QSizePolicy>

#include "util/Utils.h"

PreferencesWindow::PreferencesWindow(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::PreferencesWindow)
    , m_init(false)
{
    m_ui->setupUi(this);

    for (SettingItem::SettingMap::iterator i = SettingItem::items().begin(); i != SettingItem::items().end(); i++)
    {
        QString fullKey = i.key();
        SettingItem* setting = i.value();
        QString group = setting->group();

        if (m_settings.contains(group))
        {
            m_settings[group].append(setting);
        }
        else
        {
            QList<SettingItem*> settings;
            settings.append(setting);
            m_settings.insert(group, settings);
        }
    }

    for (QMap<QString, QList<SettingItem*>>::iterator i = m_settings.begin(); i != m_settings.end(); i++)
    {
        QString group = i.key();
        QList<SettingItem*>& settings = i.value();
        QWidget* page = new QWidget;
        m_ui->tabWidgetMain->addTab(page, group);

        QGridLayout* layout = new QGridLayout(page);

        int row = 0;
        for (QList<SettingItem*>::iterator ii = settings.begin(); ii != settings.end(); ii++)
        {
            SettingItem* item = *ii;
            QLabel* label = new QLabel(item->key());
            QWidget* widget = item->createWidget();
            QLabel* description = new QLabel(item->description());

            layout->addWidget(label, row, 0);
            layout->addWidget(widget, row, 1);
            layout->addWidget(description, row, 2);

            row++;
        }

        QSpacerItem* spacer = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
        layout->addItem(spacer, row, 2);
    }

    connect(m_ui->actionSave, &QAction::triggered, this, &PreferencesWindow::onActionSave);
}

PreferencesWindow::~PreferencesWindow()
{
}

void PreferencesWindow::onActionSave(bool checked)
{
    Settings::save();
}

void PreferencesWindow::init()
{
    
}

