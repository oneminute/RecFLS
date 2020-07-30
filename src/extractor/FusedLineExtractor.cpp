#include "FusedLineExtractor.h"

#include <opencv2/cudafilters.hpp>
#include <pcl/segmentation/impl/extract_clusters.hpp>

#include <pcl/common/pca.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/keypoints/sift_keypoint.h>



#include "util/Utils.h"
#include "util/StopWatch.h"
#include "cuda/cuda.hpp"
#include "cuda/FusedLineInternal.h"
#include "EDLines.h"
#include "device/SensorReaderDevice.h"
#include "common/Parameters.h"
#include "extractor/BoundaryExtractor.h"

bool operator<(const Eigen::Vector3f& key1, const Eigen::Vector3f& key2)
{
	if (key1.x() == key2.x())
	{
		if (key1.y() == key2.y())
		{
			return key1.z() < key2.z();
		}
		else
		{
			return key1.y() < key2.y();
		}
	}
	else
	{
		return key1.x() < key2.x();
	}

}

static std::vector<Eigen::Vector3i> QuadrantsSteps{
	Eigen::Vector3i(1, 1, 1),
	Eigen::Vector3i(-1, 1, 1),
	Eigen::Vector3i(-1, -1, 1),
	Eigen::Vector3i(1, -1, 1),
	Eigen::Vector3i(1, 1, -1),
	Eigen::Vector3i(-1, 1, -1),
	Eigen::Vector3i(-1, -1, -1),
	Eigen::Vector3i(1, -1, -1)
};

FusedLineExtractor::FusedLineExtractor(QObject* parent)
	: QObject(parent)
	, m_init(false)
	, m_resolution(0.005f)
{

}

FusedLineExtractor::~FusedLineExtractor()
{
}

void FusedLineExtractor::init(Frame& frame)
{
	if (!m_init)
	{
		cuda::Parameters params;
		params.colorWidth = frame.getColorWidth();
		params.colorHeight = frame.getColorHeight();
		params.depthWidth = frame.getDepthWidth();
		params.depthHeight = frame.getDepthHeight();

		params.cx = frame.getDevice()->cx();
		params.cy = frame.getDevice()->cy();
		params.fx = frame.getDevice()->fx();
		params.fy = frame.getDevice()->fy();
		params.minDepth = Settings::BoundaryExtractor_MinDepth.value();
		params.maxDepth = Settings::BoundaryExtractor_MaxDepth.value();
		params.borderLeft = Settings::BoundaryExtractor_BorderLeft.intValue();
		params.borderRight = Settings::BoundaryExtractor_BorderRight.intValue();
		params.borderTop = Settings::BoundaryExtractor_BorderTop.intValue();
		params.borderBottom = Settings::BoundaryExtractor_BorderBottom.intValue();
		params.depthShift = 1000;
		params.normalKernalRadius = Settings::BoundaryExtractor_CudaNormalKernalRadius.intValue();
		params.normalKnnRadius = Settings::BoundaryExtractor_CudaNormalKnnRadius.value();
		params.boundaryEstimationRadius = Settings::BoundaryExtractor_CudaBEKernalRadius.intValue();
		params.boundaryGaussianSigma = Settings::BoundaryExtractor_CudaGaussianSigma.value();
		params.boundaryGaussianRadius = Settings::BoundaryExtractor_CudaGaussianKernalRadius.intValue();
		params.boundaryEstimationDistance = Settings::BoundaryExtractor_CudaBEDistance.value();
		params.boundaryAngleThreshold = Settings::BoundaryExtractor_CudaBEAngleThreshold.value();
		params.classifyRadius = Settings::BoundaryExtractor_CudaClassifyKernalRadius.intValue();
		params.classifyDistance = Settings::BoundaryExtractor_CudaClassifyDistance.value();
		params.peakClusterTolerance = Settings::BoundaryExtractor_CudaPeakClusterTolerance.intValue();
		params.minClusterPeaks = Settings::BoundaryExtractor_CudaMinClusterPeaks.intValue();
		params.maxClusterPeaks = Settings::BoundaryExtractor_CudaMaxClusterPeaks.intValue();
		params.cornerHistSigma = Settings::BoundaryExtractor_CudaCornerHistSigma.value();


		m_frameGpu.parameters = params;
		m_frameGpu.allocate();

		m_init = true;
	}
}



FLFrame FusedLineExtractor::compute(Frame& frame)
{
	init(frame);

	FLFrame flFrame;
	flFrame.setIndex(frame.deviceFrameIndex());
	flFrame.setTimestamp(frame.timeStampColor());

	// 抽取edline直线
	cv::Mat grayImage;
	cv::cvtColor(frame.colorMat(), grayImage, cv::COLOR_RGB2GRAY);
	EDLines lineHandler = EDLines(grayImage, SOBEL_OPERATOR);
	m_linesMat = lineHandler.getLineImage();
	//cv::Mat edlinesMat = lineHandler.drawOnImage();
	m_colorLinesMat = cv::Mat(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, cv::Scalar(255, 255, 255));

	//cv::imshow("ed lines", edlinesMat);
	//cv::imshow("lines", linesMat);
	int linesCount = lineHandler.getLinesNo();
	std::vector<LS> lines = lineHandler.getLines();
	
	// 抽取出的直线集合放在这儿
	m_groupPoints.clear();
	for (int i = 0; i < linesCount; i++)
	{
		m_groupPoints.insert(i, pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>));
	}

	m_frameGpu.upload(frame.depthMat());

	// 用cuda抽取be点和折线点
	cuda::generatePointCloud(m_frameGpu);

	m_frameGpu.boundaryMat.download(m_boundaryMat);
	m_frameGpu.pointsMat.download(m_pointsMat);
	std::vector<float3> points;
	m_frameGpu.pointCloud.download(points);
	std::vector<float3> normals;
    m_frameGpu.pointCloudNormals.download(normals);
	std::vector<uchar> boundaries;
	m_frameGpu.boundaries.download(boundaries);
	std::vector<int> indicesImage;
	int cols = 0;
	m_frameGpu.indicesImage.download(indicesImage, cols);
	std::cout << "indices image cols: " << cols << std::endl;

	// 开始2d和3d的比对。
	m_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
	m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
	//m_normals.reset(new pcl::PointCloud<pcl::Normal>);
	int negativeNum = 0;
	for (int i = 0; i < frame.getDepthHeight(); i++)
	{
		for (int j = 0; j < frame.getDepthWidth(); j++)
		{
			cv::Point coord(j, i);
			int index = i * frame.getDepthWidth() + j;
			float3 value = points[index];
			uchar pointType = boundaries[index];
			ushort lineNo = m_linesMat.ptr<ushort>(i)[j];
			
			pcl::PointXYZINormal ptIn;
			//pcl::Normal normal;
			//pcl::PointNormal pn;

			//ptI.x = value.x;
			//ptI.y = value.y;
			//ptI.z = value.z;
			//ptI.intensity = lineNo;

			ptIn.x = value.x;
			ptIn.y = value.y;
			ptIn.z = value.z;
			ptIn.intensity = lineNo;

			ptIn.normal_x = normals[index].x;
			ptIn.normal_y = normals[index].y;
			ptIn.normal_z = normals[index].z;


			int ptIndex = m_pointsMat.at<int>(coord);
			if (ptIndex < 0)
			{
				negativeNum++;
			}
			else
			{
				ptIndex -= negativeNum;
				m_pointsMat.at<int>(coord) = ptIndex;
				if (!std::isnan(ptIn.x) && !std::isnan(ptIn.y) && !std::isnan(ptIn.z) 
					&& ptIn.z >= Settings::BoundaryExtractor_MinDepth.value() 
					&& ptIn.z <= Settings::BoundaryExtractor_MaxDepth.value())
				//if (indicesImage[index] > 0)
					m_cloud->points.push_back(ptIn);
			}

			//std::cout << j << ", " << i << ": " << lineNo << std::endl;

			if (pointType > 0 && lineNo != 65535)
			{
				//m_cloud->points.push_back(ptIn);
				m_allBoundary->points.push_back(ptIn);
				m_groupPoints[lineNo]->points.push_back(ptIn);
			}
		}
	}
	m_cloud->width = m_cloud->points.size();
	m_cloud->height = 1;
	m_cloud->is_dense = true;
	m_allBoundary->width = m_allBoundary->points.size();
	m_allBoundary->height = 1;
	m_allBoundary->is_dense = true;

	for (int i = 0; i < linesCount; i++)
	{
		if (m_groupPoints[i]->size() < 10)
			continue;

		Eigen::Vector3f gCenter(Eigen::Vector3f::Zero());
		for (int j = 0; j < m_groupPoints[i]->points.size(); j++)
		{
			Eigen::Vector3f np = m_groupPoints[i]->points[j].getArray3fMap();
			gCenter += np;
		}
		gCenter /= m_groupPoints[i]->points.size();

		// 因为同一直线编号的点集中可能既有真点也有veil点，所以先做区域分割。
		pcl::IndicesClusters clusters;
		pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ece;
		ece.setClusterTolerance(0.05f);
		ece.setInputCloud(m_groupPoints[i]);
		ece.setMinClusterSize(1);
		ece.setMaxClusterSize(m_groupPoints[i]->points.size());
		ece.extract(clusters);

		//std::cout << i << ": " << "count = " << m_groupPoints[i]->points.size() << std::endl;

		int maxSize = 0;
		int maxIndex = 0;
		// 分割后，找出点数最多的子区域作为初始内点集合。即cloud。
		for (int j = 0; j < clusters.size(); j++)
		{
			Eigen::Vector3f clusterCenter(Eigen::Vector3f::Zero());
			for (int n = 0; n < clusters[j].indices.size(); n++)
			{
				Eigen::Vector3f np = m_groupPoints[i]->points[clusters[j].indices[n]].getArray3fMap();
				clusterCenter += np;
			}
			clusterCenter /= clusters[j].indices.size();
			bool valid = true;
			if (clusterCenter.z() > gCenter.z())
			{
				float dist = clusterCenter.z() - gCenter.z();
				if (dist > 0.03f)
				{
					valid = false;
				}
			}
			//std::cout << "  sub " << j << ", count = " << clusters[j].indices.size() << ", farer = " << (clusterCenter.z() > gCenter.z()) << ", z dist = " << (clusterCenter.z() - gCenter.z())
				//<< ", valid = " << valid << std::endl;
			if (valid && clusters[j].indices.size() > maxSize)
			{
				maxSize = clusters[j].indices.size();
				maxIndex = j;
			}
		}
		if (maxSize < 3)
			continue;

		pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
		pcl::copyPointCloud(*m_groupPoints[i], clusters[maxIndex].indices, *cloud);
		//pcl::copyPointCloud(*m_groupPoints[i], *cloud);

		// 计算这个初始内点集合的主方向和中点。
		pcl::PCA<pcl::PointXYZINormal> pca;
		pca.setInputCloud(cloud);
		Eigen::Vector3f eigenValues = pca.getEigenValues();
		float sqrt1 = sqrt(eigenValues[0]);
		float sqrt2 = sqrt(eigenValues[1]);
		float sqrt3 = sqrt(eigenValues[2]);
		float a1 = (sqrt1 - sqrt2) / sqrt1;
		float a2 = (sqrt2 - sqrt3) / sqrt1;
		float a3 = sqrt3 / sqrt1;

		//std::cout << "  " << m_groupPoints[i]->size() << ", cluster: " << clusters.size() << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3 << std::endl;
		//std::cout << "  init inliers size: " << cloud->size() << std::endl;

		// 主方向
		Eigen::Vector3f dir = pca.getEigenVectors().col(0).normalized();
		// 中点
		Eigen::Vector3f center = pca.getMean().head(3);

		// 然后，遍历剩余的子区域点集，查看每一个点到这条直线的距离是否在阈值以内，在就加到内点集中，不在就抛弃。
		for (int j = 0; j < clusters.size(); j++)
		{
			if (j == maxIndex)
				continue;

			for (int n = 0; n < clusters[j].indices.size(); n++)
			{
				int nIndex = clusters[j].indices[n];
				pcl::PointXYZINormal pclPt = m_groupPoints[i]->points[nIndex];
				Eigen::Vector3f pt = pclPt.getArray3fMap();
				float dist = (pt - center).cross(dir).norm();
				// 暂时硬编码的阈值。
				if (dist <= 0.05f)
				{
					cloud->points.push_back(pclPt);
				}
			}
		}
		if (cloud->size() < 10)
			continue;

		// 最后再计算一遍内点集的主方向与中点。
		//std::cout << "    final: " << cloud->size() << ", max size: " << maxSize << ", max index: " << maxIndex << std::endl;
		//std::cout << "    final: " << cloud->size() << std::endl;
		pcl::PCA<pcl::PointXYZINormal> pcaFinal;
		pcaFinal.setInputCloud(cloud);
		eigenValues = pcaFinal.getEigenValues();
		dir = pcaFinal.getEigenVectors().col(0).normalized();
		center = pcaFinal.getMean().head(3);
		//Eigen::Vector3f eigenValues = pcaFinal.getEigenValues();
		//Eigen::Vector3f dir = pcaFinal.getEigenVectors().col(0).normalized();
		//Eigen::Vector3f center = pcaFinal.getMean().head(3);

		// 确定端点。
		Eigen::Vector3f start(0, 0, 0);
		Eigen::Vector3f end(0, 0, 0);
		Eigen::Vector3f avgNormal(0, 0, 0);
		for (int j = 0; j < cloud->size(); j++)
		{
			pcl::PointXYZINormal& ptBoundary = cloud->points[j];
			Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
			Eigen::Vector3f projPoint = closedPointOnLine(boundaryPoint, dir, center);
			avgNormal += ptBoundary.getNormalVector3fMap();

			if (start.isZero())
			{
				// 如果第一次循环，让当前点作为起点
				start = projPoint;
			}
			else
			{
				// 将当前点与当前计算出的临时起点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的起点。
				if ((start - projPoint).dot(dir) > 0)
				{
					start = projPoint;
				}
			}

			if (end.isZero())
			{
				// 如果第一次循环，让当前点作为终点
				end = projPoint;
			}
			else
			{
				// 将当前点与当前计算出的临时终点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的终点。
				if ((projPoint - end).dot(dir) > 0)
				{
					end = projPoint;
				}
			}
		}
		avgNormal /= cloud->size();

		LS line2d = lines[i];
		LineSegment line;
		line.setStart(start);
		line.setEnd(end);
		line.setSecondaryDir(avgNormal);
		line.setStart2d(line2d.start);
		line.setEnd2d(line2d.end);
		line.calculateColorAvg(frame.colorMat());
		line.drawColorLine(m_colorLinesMat);
		//line.reproject();
		//std::cout << line.shortDescriptorSize() << std::endl;
		line.setIndex(flFrame.lines()->points.size());
		if (line.length() > 0.1f)
		{
			//m_lines.insert(i, line);
			flFrame.lines()->points.push_back(line);
		}
		flFrame.lines()->width = flFrame.lines()->points.size();
		flFrame.lines()->height = 1;
		flFrame.lines()->is_dense = true;
	}

	//qDebug() << "all boundary points:"/ << m_allBoundary->size();
	//TOCK("boundaries_downloading");
	//linefilter(flFrame.lines());
	//flFrame.lines()->width = flFrame.lines()->points.size();

	return flFrame;
}

void FusedLineExtractor::linefilter(pcl::PointCloud<LineSegment>::Ptr lines)
{
	std::sort(lines->points.begin(), lines->points.end(), [](const LineSegment& l1, const LineSegment& l2) -> bool 
	{
		return l1.length() > l2.length();
	}
	);

	/*for (int i = 0; i < lines->points.size(); i++)
	{
		std::cout << i << ": " << lines->points[i].length() << std::endl;
	}*/

	size_t remains = std::max(static_cast<size_t>(20), static_cast<size_t>(lines->points.size() * 0.2));
	remains = std::min(remains, lines->points.size());

	std::cout << "before resize: " << lines->points.size() << std::endl;
	lines->points.resize(remains);
	std::cout << "after resize: " << lines->points.size() << std::endl;
}


void FusedLineExtractor::generateCylinderDescriptors(pcl::PointCloud<LineSegment>::Ptr lines, float radius, int segments, int angleSegments, float width, float height, float cx, float cy, float fx, float fy)
{
	cv::Mat board(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < lines->points.size(); i++)
	{
		LineSegment ls = lines->points[i];
		cv::Point2f cvDir2d = ls.end2d() - ls.start2d();
		Eigen::Vector3f vDir(cvDir2d.y, -cvDir2d.x, 0);
		vDir.normalize();
		Eigen::Vector3f pa = ls.start() - vDir * radius;
		Eigen::Vector3f pb = ls.start() + vDir * radius;
		Eigen::Vector3f pc = ls.end() + vDir * radius;
		Eigen::Vector3f pd = ls.end() - vDir * radius;
		std::vector<cv::Point2f> contours(4);
		contours[0] = cv::Point2f(pa.x() * fx / pa.z() + cx, pa.y() * fy / pa.z() + cy);
		contours[1] = cv::Point2f(pb.x() * fx / pb.z() + cx, pb.y() * fy / pb.z() + cy);
		contours[2] = cv::Point2f(pc.x() * fx / pc.z() + cx, pc.y() * fy / pc.z() + cy);
		contours[3] = cv::Point2f(pd.x() * fx / pd.z() + cx, pd.y() * fy / pd.z() + cy);
		//contours[4] = contours[0];
		cv::Rect rect = cv::boundingRect(contours);

		int histSize = segments * angleSegments;
		std::vector<Eigen::Vector3f> neighbours;
		std::vector<int> hist(histSize);
		//std::vector<int> tmpHist(histSize);
		for (int n = 0; n < histSize; n++)
		{
			hist[n] = 0;
			//tmpHist[n] = 0;
		}

		std::vector<int> angles(angleSegments);
		for (int n = 0; n < angleSegments; n++)
		{
			angles[n] = 0;
		}

		int maxAngleIndex = 0;
		int maxAngleCount = 0;
		for (int ix = 0; ix < rect.width; ix++)
		{
			for (int iy = 0; iy < rect.height; iy++)
			{
				cv::Point pixel(rect.x + ix, rect.y + iy);
				if (pixel.x >= 0 && pixel.x < width && pixel.y >= 0 && pixel.y < height)
				{
					if (cv::pointPolygonTest(contours, pixel, false) >= 0)
					{
						board.at<cv::Vec3b>(pixel) = cv::Vec3b(0, 255, 255);
						float3 point3 = m_points[pixel.y * width + pixel.x];
						Eigen::Vector3f point(point3.x, point3.y, point3.z);
						Eigen::Vector3f edge = point - ls.start();
						Eigen::Vector3f dir = ls.direction().normalized();
						float dist = edge.cross(dir).norm();
						if (dist <= radius)
						{
							neighbours.push_back(point);
							Eigen::Vector3f vert = edge.cross(dir).cross(dir).normalized();
							float radians = qAtan2(vDir.dot(vert), dir.dot(vert)) + M_PI;
							int angleSeg = radians * angleSegments / (M_PI * 2);
							int distSeg = qFloor(dist * segments / radius);
							angles[angleSeg]++;
							hist[angleSeg * segments + distSeg]++;
							if (angles[angleSeg] > maxAngleCount)
							{
								maxAngleIndex = angleSeg;
								maxAngleCount = angles[angleSeg];
							}
						}
					}
				}
			}
		}

		if (neighbours.size() >= 3)
		{
			for (int n = 0; n < histSize; n++)
			{
				int newIndex = (n + histSize - maxAngleIndex * segments) % histSize;
				int tmp = hist[newIndex];
				hist[newIndex] = hist[n];
				hist[n] = tmp;
			}
		}

		for (int n = 0; n < histSize; n++)
		{
			std::cout << std::setw(4) << hist[n] << " ";
		}
		std::cout << std::endl;
		//if (i == 0)
		{
			cv::line(board, ls.start2d(), ls.end2d(), cv::Scalar(255, 0, 0));
			std::vector<std::vector<cv::Point2f>> cc;
			cc.push_back(contours);
			cv::line(board, contours[0], contours[1], cv::Scalar(0, 0, 255));
			cv::line(board, contours[1], contours[2], cv::Scalar(0, 0, 255));
			cv::line(board, contours[2], contours[3], cv::Scalar(0, 0, 255));
			cv::line(board, contours[3], contours[0], cv::Scalar(0, 0, 255));
			cv::rectangle(board, rect, cv::Scalar(0, 255, 0));
		}
	}
	cv::imwrite("desc01.png", board);
}



void FusedLineExtractor::generateVoxelsDescriptors(Frame& frame, pcl::PointCloud<LineSegment>::Ptr lines, float radius, int radiusSegments, int segments, int angleSegments, float width, float height, float cx, float cy, float fx, float fy)
{
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZINormal> octree(m_resolution);
	octree.setInputCloud(m_allBoundary);
	octree.addPointsFromInputCloud();

	double minX, minY, minZ, maxX, maxY, maxZ;
	octree.getBoundingBox(minX, minY, minZ, maxX, maxY, maxZ);
	Eigen::Vector3f minPoint(minX, minY, minZ);
	Eigen::Vector3f maxPoint(maxX, maxY, maxZ);
	std::cout << "voxel dims: " << qFloor((maxX - minX) / m_resolution) << ", " << qFloor((maxY - minY) / m_resolution) << ", " << qFloor((maxZ - minZ) / m_resolution) << std::endl;

	int leafCount = octree.getLeafCount();
	int branchCount = octree.getBranchCount();

	std::cout << "[BoundaryExtractor::computeVBRG] branchCount:" << branchCount << ", leafCount:" << leafCount << std::endl;
	//std::cout << "[BoundaryExtractor::computeVBRG] bounding box:" << minX << minY << minZ << maxX << maxY << maxZ << std::endl;

	pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it(&octree);

	// 在原点处生成一个圆，半径为radius。
	int voxelRadius = qFloor(radius / m_resolution);
	std::vector<std::vector<Eigen::Vector3f>> baseCircles(radiusSegments);
	for (int i = 0; i < radiusSegments; i++)
	{
		baseCircles[i] = std::vector<Eigen::Vector3f>();
	}

	for (int r = -voxelRadius; r <= voxelRadius; r++)
	{
		for (int c = -voxelRadius; c <= voxelRadius; c++)
		{
			float radius = qSqrt(r * 1.f * r + c * 1.f * c);
			if (radius >= voxelRadius)
			{
				std::cout << "  ";
				continue;
			}

			int circleIndex = qFloor(radius * radiusSegments / voxelRadius);
			baseCircles[circleIndex].push_back(Eigen::Vector3f(c, r, 0));
			//std::cout << circleIndex << ". [" << c << ", " << r << ", " << 0 << "]" << std::endl;
			std::cout << circleIndex << " ";
		}
		std::cout << std::endl;
	}

	/*for (int i = 0; i < baseCircles.size(); i++)
	{
		std::cout << i << ". ";
		for (int j = 0; j < baseCircles[i].size(); j++)
		{
			Eigen::Vector3f key = baseCircles[i][j];
			std::cout << "[" << key.x() << ", " << key.y() << ", " << key.z() << "] ";
		}
		std::cout << std::endl;
	}*/

	for (int i = 0; i < lines->points.size(); i++)
	{
		LineSegment& ls = lines->points[i];
		Eigen::Vector3f start = ls.start();
		Eigen::Vector3f end = ls.end();

		Eigen::Vector3f startKey = ((start - minPoint) / m_resolution);
		Eigen::Vector3f endKey = ((end - minPoint) / m_resolution);

		std::cout << i << ". [" << start.transpose() << "] [" << end.transpose() << "] [" << startKey.transpose() << "] [" << endKey.transpose() << "]" << std::endl;
		Eigen::Vector3f voxelDir = endKey - startKey;
		Eigen::Vector3f voxelDirN = voxelDir.normalized();
		//Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitY(), voxelDir);
		Eigen::Quaternionf q(ls.localRotaion());
		Eigen::Quaternionf qI = q.inverse();

		QMap<Eigen::Vector3f, bool> done;
		int steps = qFloor(voxelDir.norm());

		// calculate cylinders layer by layer.
		std::vector<std::vector<Eigen::Vector3f>> lineCylinders(baseCircles.size());
		Eigen::Vector3f currentKey = startKey;
		for (int si = 0; si < steps; si++)
		{
			currentKey += voxelDirN * si;
			if (done.contains(currentKey))
			{
				continue;
			}
			done.insert(currentKey, true);

			for (int ci = 0; ci < baseCircles.size(); ci++)
			{
				for (int ki = 0; ki < baseCircles[ci].size(); ki++)
				{
					Eigen::Vector3f key = q * baseCircles[ci][ki];
					key += currentKey;
					if (octree.existLeaf(key.x(), key.y(), key.z()))
						lineCylinders[ci].push_back(key);
				}
			}
		}
		// end of calculating
		
		std::vector<float> desc(radiusSegments * 8);
		for (int ci = 0; ci < baseCircles.size(); ci++)
		{
			std::vector<float> layerQuadrants{ 0, 0, 0, 0, 0, 0, 0, 0 };

			Eigen::Vector3f lastKey = startKey;
			Eigen::Vector3f pKey = startKey;
			for (int li = 0; li <= steps; li++)
			{
				pKey += voxelDirN * li;	// line voxel
				if (li != 0)
				{
					if ((pKey - lastKey).squaredNorm() < 1)
						continue;
				}
				lastKey = pKey;

				// fetch normal
				pcl::Indices voxelIndices;
				pcl::PointXYZINormal ptCenter;
				Eigen::Vector3f ePtCenter = minPoint + pKey * m_resolution;
				ptCenter.getVector3fMap() = ePtCenter;
				//int count = octree.voxelSearch(ptCenter, voxelIndices);
				//int count = octree.getIntersectedVoxelIndices(pKey, voxelDirN, voxelIndices);
				int nearestIndex;
				float nearstDist;
				octree.approxNearestSearch(ptCenter, nearestIndex, nearstDist);
				Eigen::Vector3f pNormal = m_cloud->points[nearestIndex].getNormalVector3fMap();
				//if (count == 0)
				//{
				//	continue;
				//}
				//Eigen::Vector3f pNormal(Eigen::Vector3f::Zero());
				//// use average of all normals of the voxel as final normal.
				//for (int ni = 0; ni < voxelIndices.size(); ni++)
				//{
				//	int index = voxelIndices[ni];
				//	pcl::PointXYZINormal pt = m_allBoundary->points[index];
				//	pNormal += pt.getNormalVector3fMap();
				//}
				//pNormal /= count;

				int quadrant = -1;
				Eigen::Vector4f quaternion(Eigen::Vector4f::Zero());
				std::vector<float> quadrants{ 0, 0, 0, 0, 0, 0, 0, 0 };
				for (int cyi = 0; cyi < lineCylinders[ci].size(); cyi++)
				{
					// calculate the quaternion
					Eigen::Vector3f key = lineCylinders[ci][cyi];	// neighbour voxel
					int x = floor(key.x());
					int y = floor(key.y());
					int z = floor(key.z());
					if (!octree.existLeaf(x, y, z))
					{
						continue;
					}

					Eigen::Vector3f PN = key - pKey;

					float m = PN.norm() * m_resolution;
					float theta = atan((key.y() - pKey.y()) / (key.z() - pKey.z()));
					float phi = asin(key.z() - pKey.z()) / m;
					float delta = acos(PN.dot(pNormal) / PN.norm());
					quaternion.w() = m;
					quaternion.x() = theta;
					quaternion.y() = phi;
					quaternion.z() = delta;

					Eigen::Vector3f vec(theta, phi, delta);
					vec = qI * vec;

					if (vec.x() >= 0)
					{
						if (vec.y() >= 0)
						{
							if (vec.z() >= 0)
								quadrant = 0;
							else
								quadrant = 1;
						}
						else
						{
							if (vec.z() >= 0)
								quadrant = 2;
							else
								quadrant = 3;
						}
					}
					else
					{
						if (vec.y() >= 0)
						{
							if (vec.z() >= 0)
								quadrant = 4;
							else
								quadrant = 5;
						}
						else
						{
							if (vec.z() >= 0)
								quadrant = 6;
							else
								quadrant = 7;
						}
					}

					if (quadrant >= 0)
					{
						quadrants[quadrant] += m;
					}
				}
				int maxIndex = std::distance(quadrants.begin(), std::max_element(quadrants.begin(), quadrants.end()));
				float maxM = quadrants[maxIndex];

				layerQuadrants[maxIndex] += maxM;
			}
			for (int n = 0; n < layerQuadrants.size(); n++)
			{
				std::cout << std::setw(12) << layerQuadrants[n];
				desc[ci * 8 + n] = layerQuadrants[n];
			}
		}
		
		ls.setLongDescriptor(desc);
		std::cout << std::endl;
	}

}

//int FusedLineExtractor::quadrantStatisticByVoxel(pcl::octree::OctreePointCloudSearch<pcl::PointXYZINormal>& tree, const Eigen::Vector3f& key, int length, int xStep, int yStep, int zStep)
//{
//	int total = 0;
//	for (int i = 0; i < length; i++)
//	{
//		for (int j = 0; j < length; j++)
//		{
//			for (int k = 0; k < length; k++)
//			{
//				// [x, y, z] represents a neighbour coordinate.
//				int x = qFloor(key.x() + i * xStep);
//				int y = qFloor(key.y() + i * yStep);
//				int z = qFloor(key.z() + i * zStep);
//
//				// we should fetch all points of the current neighbour voxel.
//				if (tree.existLeaf(x, y, z))
//				{
//					
//					//total++;
//				}
//				
//				Cvec vcenter = { key.x(), key.y(), key.z() };
//				Cvec vdisp, grad;
//
//				vdisp.x = (x - vcenter.x) * m_resolution;
//				vdisp.y = (y - vcenter.y) * m_resolution;
//				vdisp.z = (z - vcenter.z) * m_resolution;
//				
//				
//				grad.x = 0.5 * ((x + 1, y, z) - (x - 1, y, z));
//				grad.y = 0.5 * ((x, y + 1, z) - (x, y - 1, z));
//				grad.z = 0.5 * ((x, y, z + 1) - (x, y, z - 1));
//
//				grad.x *= 1.0f / m_resolution;
//				grad.y *= 1.0f / m_resolution;
//				grad.z *= 1.0f / m_resolution;
//
//
//				if (tree.existLeaf(x, y, z))
//				{
//					total++;
//				}
//			}
//		}
//	}
//	return total;
//}
	







