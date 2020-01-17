#ifndef DDBPLINEMATCHER_H
#define DDBPLINEMATCHER_H

#include <QObject>

class DDBPLineMatcher : public QObject
{
    Q_OBJECT
public:
    explicit DDBPLineMatcher(QObject* parent = nullptr);
};

#endif // DDBPLINEMATCHER_H
