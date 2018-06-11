// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Sample of using OpenCV dnn module with Tensorflow Inception model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>

#include "mnist_reader_less.hpp"


wchar_t*	M2W(const char *instr)
{
	static wchar_t* dest = NULL;

	if ( ( dest != NULL ) || ( instr == NULL ) )
	{
		delete[] dest;
		dest = NULL;
	}

	if ( instr == NULL )
	{
		return NULL;
	}

	int len = strlen(instr) + 1;
	dest = (wchar_t*)calloc(len * 2,1);
	MultiByteToWideChar(CP_ACP,0,instr,-1,dest,len);

	return dest;
}
char*	W2M(const wchar_t *instr)
{
	static char* dest = NULL;

	if ( ( dest != NULL ) || ( instr == NULL ) )
	{
		delete[] dest;
		dest = NULL;
	}

	if ( instr == NULL )
	{
		return NULL;
	}

	int len = WideCharToMultiByte( CP_ACP, 0, instr, -1, NULL, 0, NULL, NULL );
	dest = (char*)calloc(len+1,1);

	WideCharToMultiByte(CP_ACP,0,instr,-1,dest,len,NULL,NULL);

	return dest;

}
//using namespace std;
void read_directory(const std::string& name, std::vector<std::string> & v)
{
	//String pattern(name);
	std::string pattern(name);
	std::string curpath(name);
	curpath.append("\\");

	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	//wchar_t	wc[MAX_PATH] = L"";
	//char mc[MAX_PATH] = "";
	//mbtowc( wc, pattern.c_str(), pattern.length() );
	char ext[100] = "";
	char fname[100]= "";
	if ((hFind = FindFirstFile(M2W( pattern.c_str()), &data)) != INVALID_HANDLE_VALUE) {
		do {
			//wctomb_s( mc,data.cFileName );
			std::string filename = W2M( data.cFileName );

			
			_splitpath( filename.c_str(), NULL, NULL, fname, ext);			
			if( !strcmp(ext,".jpg") || 
				!strcmp(ext,".png") )
			{				
				v.push_back(curpath + std::string(W2M(data.cFileName)));
			}
			
			
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}


using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;


const String keys =
	"{help h    || Sample app for loading Inception TensorFlow model. "
	"The model and class names list can be downloaded here: "
	"https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip }"
	"{model m   |frozen_CNN_final.pb| path to TensorFlow .pb model file }"
	"{image i   |cat.jpg| path to image file }"
	"{i_blob    | input | input blob name) }"
	"{o_blob    | pred_model | output blob name) }"
	"{c_names c | mnist_labels_test.txt | path to file with classnames for class id }"
	"{result r  || path to save output blob (optional, binary format, NCHW order) }"
	;

//const String keys =
//	"{help h    || Sample app for loading Inception TensorFlow model. "
//	"The model and class names list can be downloaded here: "
//	"https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip }"
//	"{model m   |tensorflow_inception_graph.pb| path to TensorFlow .pb model file }"
//	"{image i   |cat.jpg| path to image file }"
//	"{i_blob    | Input | input blob name) }"
//	"{o_blob    | Outmodel | output blob name) }"
//	"{c_names c | mnist_labels_test.txt | path to file with classnames for class id }"
//	"{result r  || path to save output blob (optional, binary format, NCHW order) }"
//	;

void getMaxClass(const Mat &probBlob, int *classId, double *classProb);
std::vector<String> readClassNames(const char *filename);
int TensorflowImportTest(int argc, char *argv[])
{
	MNISTDataset mnist;
	//assert(0 == mnist.Parse(".\\data\\t10k-images-idx3-ubyte.gz",".\\data\\t10k-labels-idx1-ubyte.gz"));
	mnist.Parse(".\\data\\t10k-images.idx3-ubyte",".\\data\\t10k-labels.idx1-ubyte");
	//mnist.Print();

	const\
	float	*fimgbuf		=	mnist.GetImageData();
	size_t	imgsize			=	mnist.GetImageCount();
	size_t	height			=	mnist.GetImageHeight();
	size_t	width			=	mnist.GetImageWidth();
	const\
	uint8_t *labels			=	mnist.GetCategoryData();

	printf("%dX%d -> [#%d]\n",
		width, height, imgsize );
	float	*fbuff			=	(float*)fimgbuf;
	size_t	buffsize		=	width * height;

	//mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
	//mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);


	std::vector<std::string> flist;
	std::string name = ".\\img";
	read_directory(name, flist);
	cv::CommandLineParser parser(argc, argv, keys);



	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String modelFile = parser.get<String>("model");
	String imageFile = parser.get<String>("image");	
	String inBlobName = parser.get<String>("i_blob");
	String outBlobName = parser.get<String>("o_blob");

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	String classNamesFile = parser.get<String>("c_names");
	String resultFile = parser.get<String>("result");
	cv::TickMeter tm;
	//! [Initialize network]
	tm.start();
	dnn::Net net = readNetFromTensorflow(modelFile);
	tm.stop();
	std::cout << "read net from tensorflow, ms: " << tm.getTimeMilli()  << std::endl;

	tm.start();

	Sleep(1000);// 1000 mili sec. delays
	tm.stop();
	std::cout<<"nothing"<<tm.getTimeMilli()<<endl;
	//! [Initialize network]


	if (net.empty())
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << modelFile << std::endl;		
		exit(-1);
	}

	int iter = 0;
	double elapsetime = tm.getTimeMilli();
	int maxiter = 10;
	double avgttact = 0;

	while(iter++<maxiter)
	{
		int id = rand() % flist.size();
		//Mat img = imread(flist[id]);
		Mat img = cv::Mat(height,width,CV_32FC1, fbuff );
		cv::Mat normimg;
		cv::normalize( img, normimg, 0, 1, cv::NORM_MINMAX  );
		
		fbuff += buffsize;
		if (img.empty())
		{
			std::cerr << "Can't read image from the file: " << imageFile << std::endl;
			continue;
			//exit(-1);
		}
		tm.start();
		//Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224), Scalar(), true, false);   //Convert Mat to batch of images
		
		Mat inputBlob = blobFromImage(normimg, 1.0f, Size(28, 28), Scalar(), true, false);   //Convert Mat to batch of images
		cout<<inputBlob.size<<endl;
		//! [Prepare blob]
		//inputBlob -= 117.0;
		//! [Set input blob]
		
		net.setInput(inputBlob, inBlobName); 
#if 0 
		std::vector<String> alllayer = net.getLayerNames();
		printf("%s\n",net.getLayerNames().back().c_str() );
		for(int k = 0; k< alllayer.size(); k++)
		{
			String layername =  alllayer[k].c_str();
			int id  = net.getLayerId(layername) ;
			printf("%d = %s, id num = %d \n", 
				k, layername.c_str() , id);

			cv::dnn::Net::LayerId lid(layername);

			Ptr<Layer> cul = net.getLayer( lid );
			std::vector<Mat> blob = cul->blobs;
			for( int k = 0; k < blob.size(); k++)
			{		
				//cout<<blob[k].size<<endl;
				//cout<<blob[k]<<endl;
			}
		}
#endif
		printf("--------------[%03d iteration]--------------\n",iter);

		//! [Make forward pass]
		
		Mat result = net.forward(outBlobName);                          //compute output
		//! [Make forward pass]

		tm.stop();


		if (!resultFile.empty()) {
			CV_Assert(result.isContinuous());

			ofstream fout(resultFile.c_str(), ios::out | ios::binary);
			fout.write((char*)result.data, result.total() * sizeof(float));
			fout.close();
		}
		double dt = tm.getTimeMilli() - elapsetime ;
		//std::cout << "Output blob shape " << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << std::endl;
		std::cout << "Inference time, ms: " << dt  << std::endl;
		elapsetime = tm.getTimeMilli();
		if( iter != 1 )
			avgttact += dt;
		if (!classNamesFile.empty()) {
			std::vector<String> classNames = readClassNames(classNamesFile.c_str());

			int classId;
			double classProb;
			getMaxClass(result, &classId, &classProb);//find the best class
			//std::cout<<result;

			//! [Print results]
			std::cout << "Best class: #" << classId << " '" <<endl;
			std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
		}
		cv::imshow("", img);
		cv::waitKey(0);
	}
	std::cout << "average tact : " << avgttact / ( maxiter - 1) << std::endl;
	system("pause");

}
int main(int argc, char **argv)
{
	TensorflowImportTest(argc, argv);
	return 0;
} //main


/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename)
{
	std::vector<String> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back( name );
	}

	fp.close();
	return classNames;
}
