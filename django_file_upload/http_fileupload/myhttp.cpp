#include "myhttp.h"
#include "ui_myhttp.h"
#include <QDebug>
#include <QTime>
#include <QHttpPart>
#include <QHttpMultiPart>>
#include <QFile>
#include <QBuffer>
#include <QMessageBox>
//#include <QRandomGenerator>
myHttp::myHttp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::myHttp)
{
    ui->setupUi(this);
    request.setUrl(QUrl("http://192.168.25.9:800"));

    manager = new QNetworkAccessManager();
//    this->addAction(ui->httpRequestbutton);
    connect(ui->httpRequestbutton,SIGNAL(clicked(bool)),this,SLOT(httprequestclicked()));
    connect(ui->TransmitImgButton,SIGNAL(clicked(bool)),this,SLOT(imgtranbtnclicked()));

    connect(manager, SIGNAL(finished(QNetworkReply*)),
        this, SLOT(managerFinished(QNetworkReply*)));


}
void myHttp::onError(QNetworkReply::NetworkError err)
{
    ui->httpresponstext->setText("cannot connect");


}
void myHttp::imgtranbtnclicked()
{
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
        QHttpPart imagePart;
        //imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("text/plain"));
        imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"file\"; filename=\"image\""));/* version.tkt is the name on my Disk of the file that I want to upload */

        QHttpPart textPart;
        textPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"name\""));
        textPart.setBody("toto");/* toto is the name I give to my file in the server */

        QString apkLocation = "d:/yuna.jpg";
        QFile *file = new QFile(apkLocation);
        file->open(QIODevice::ReadOnly);
        imagePart.setBodyDevice(file);
        file->setParent(multiPart); // we cannot delete the file now, so delete it with the multiPart

        multiPart->append(textPart);
        multiPart->append(imagePart);

        manager->post(request, multiPart);

}

void myHttp::httprequestclicked()
{

    QNetworkReply *reply =  manager->get(request);

    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(onError(QNetworkReply::NetworkError)));

    if( reply->error())
    {
        qDebug()<<reply->errorString();
    }
    else
    {

    }


}

void myHttp::managerFinished(QNetworkReply *reply) {
    qDebug()<<__FUNCTION__;
    if (reply->error()) {
        qDebug() << reply->errorString();
        ui->httpresponstext->setText("cannot connect from server");
        return;
    }
    else
    {
        QString answer = reply->readAll();
        ui->httpresponstext->setText(answer);
    }

}

myHttp::~myHttp()
{
    delete ui;
}
