#ifndef MYHTTP_H
#define MYHTTP_H

#include <QMainWindow>
#include <QProgressDialog>
#include <QNetworkAccessManager>
#include <QUrl>
#include <QNetworkRequest>
#include <QNetworkReply>
//QT_BEGIN_NAMESPACE
//class QFile;
//class QLabel;
//class QLineEdit;
//class QPushButton;
//class QSslError;
//class QAuthenticator;
//class QNetworkReply;
//class QCheckBox;
//class QNetworkRequest;
//QT_END_NAMESPACE

namespace Ui {
class myHttp;
}

class myHttp : public QMainWindow
{
    Q_OBJECT

public:
    explicit myHttp(QWidget *parent = 0);
    ~myHttp();
private slots:
    void httprequestclicked();
     void managerFinished(QNetworkReply *reply);
     void onError(QNetworkReply::NetworkError);
     void imgtranbtnclicked();
private:

    QNetworkAccessManager *manager;
    QNetworkRequest request;

    Ui::myHttp *ui;
};

#endif // MYHTTP_H
