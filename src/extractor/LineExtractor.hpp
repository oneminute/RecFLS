#include "LineExtractor.h"
#include "util/Utils.h"

#include <QDebug>
#include <QtMath>
#include <QQueue>
#include <pcl/common/pca.h>

bool LineCompare(const LineSegment& l1, const LineSegment& l2)
{
    return l1.length() > l2.length();
}

template <typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::compute(const pcl::PointCloud<PointInT> &cloudIn, pcl::PointCloud<PointOutT> &cloudOut)
{
    for (int i = 0; i < cloudIn.points.size(); i++)
    {
        pcl::PointXYZI pt;
        pt.x = cloudIn.points[i].x;
        pt.y = cloudIn.points[i].y;
        pt.z = cloudIn.points[i].z;
        pt.intensity = 0;
        boundary_->push_back(pt);
    }

    joinSortedPoints();

    for (int i = 0; i < segments_.size(); i++)
    {
        extractLinesFromSegment(segments_[i], i);
    }

    mergeCollinearLines();

    linesSortingByLength(lines_);

    createLinesTree(lines_);
    extracLinesClusters();

    linesSortingByLength(mergedLines_);

//    float minLength = mergedLines_.begin()->length();
//    float maxLength = mergedLines_.end()->length();
//    for (std::vector<LineSegment>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
//    {
//        qDebug() << i->length();
//    }

    // 重新为每一个线段的端点的intensity设值，值为线段在集合中的索引下标
//    int number = 0;
//    std::vector<LineSegment> tmpLines;
//    for (std::vector<LineSegment>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
//    {
//        if (i->length() > line_length_threshold_)
//        {
//            i->setSegmentNo(number);

//            unifyLineDirection(*i);

//            i->start.intensity = i->end.intensity = number;
            //i->generateSimpleDescriptor(minLength, maxLength);
//            tmpLines.push_back(*i);
//            number++;
//        }
//    }
//    mergedLines_ = tmpLines;

//    generateLineCloud();

//    pcl::PointXYZI minPoint, maxPoint;
//    pcl::getMinMax3D<pcl::PointXYZI>(*lineCloud_, minPoint, maxPoint);
//    pcl::Vector3fMap minValue = minPoint.getVector3fMap(), maxValue = maxPoint.getVector3fMap();
//    for (std::vector<LineSegment>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
//    {
//        i->generateShotDescriptor(minLength, maxLength, minValue, maxValue);
//        computeDescriptorFeature(*i);
//    }

}

template<typename PointInT, typename PointOutT>
std::map<int, int> LineExtractor<PointInT, PointOutT>::linesCompare(const std::vector<LineSegment> &srcLines)
{
    Eigen::MatrixXf mDesc1;
    Eigen::MatrixXf mDesc2;

    mDesc1.resize(srcLines.size(), 13);
    mDesc2.resize(mergedLines_.size(), 13);

    for (int i = 0; i < srcLines.size(); i++)
    {
        mDesc1.row(i) = srcLines[i].shortDescriptor();
    }

    for (int i = 0; i < mergedLines_.size(); i++)
    {
        mDesc2.row(i) = mergedLines_[i].shortDescriptor();
    }

    Eigen::MatrixXf m = mDesc1 * mDesc2.transpose();
    //Eigen::EigenSolver<Eigen::MatrixXf> es(m);
    //Eigen::MatrixXf ev = es.pseudoEigenvectors();
    std::cout << "samples: " << srcLines.size() << ", " << mergedLines_.size() << " m: " << m.rows() << ", " << m.cols() << std::endl;

    std::map<int, int> pairs;
    for (int i = 0; i < m.rows(); i++)
    {
        int maxIndex = -1;
        float max = 0;
        max = m.row(i).maxCoeff(&maxIndex);

        std::vector<int> candidateLineIndices;
        for (int j = 0; j < m.cols(); j++)
        {
            if (m.row(i)[j] >= 0.985f)
            {
                candidateLineIndices.push_back(j);
            }
        }

        Eigen::MatrixXf candidateM;
        candidateM.resize(candidateLineIndices.size(), srcLines[i].longDescriptor().size());
        for (int j = 0; j < candidateLineIndices.size(); j++)
        {
            candidateM.row(j) = mergedLines_[candidateLineIndices[j]].longDescriptor;
        }
        std::cout << i << "(rough) --> max: " << maxIndex << ", " << max << ", " << candidateLineIndices.size() << std::endl;

        if (candidateLineIndices.size() > 0)
        {
            Eigen::VectorXf longResult = srcLines[i].longDescriptor().transpose() * candidateM.transpose();
            std::cout << "long result size: " << longResult.size() << std::endl;
            max = longResult.maxCoeff(&maxIndex);
            maxIndex = candidateLineIndices[maxIndex];
        }
        else
        {
            max = 0;
            maxIndex = -1;
        }
        std::cout << i << "(fine) --> max: " << maxIndex << ", " << max << std::endl;

        //if (max >= 0.99f)
        if (maxIndex >= 0 && max >= 0.8f)
        {
            pairs.insert(std::make_pair(i, maxIndex));
            std::cout << i << " --> max: " << maxIndex << ", " << max << ", " << candidateLineIndices.size() << std::endl;
        }
    }

    return pairs;
}

//template<typename PointInT, typename PointOutT>
//void LineExtractor<PointInT, PointOutT>::addPointToSortedIndices(const PointInT &pt)
//{
//    pcl::PointXYZI boundaryPoint;
//    boundaryPoint.x = pt.x;
//    boundaryPoint.y = pt.y;
//    boundaryPoint.z = pt.z;
//    boundaryPoint.intensity = pt.curvature;
//    if (boundary_->points.empty())
//    {
//        boundary_->points.push_back(boundaryPoint);
//    }
//    else
//    {
//        // 二分查找
//        int startIndex = 0;
//        int endIndex = boundary_->size() - 1;
//        int targetIndex = -1;
//        while (true)
//        {
//            int middleIndex = (startIndex + endIndex) / 2;
//            int length = endIndex - startIndex + 1;
//            if (length <= 1)
//            {
//                targetIndex = middleIndex;
//                break;
//            }
//            pcl::PointXYZI middlePoint = boundary_->points[middleIndex];
//            float middleCurvature = middlePoint.intensity;
//            if (boundaryPoint.intensity < middleCurvature)
//            {
//                endIndex = middleIndex - 1;
//            }
//            else if (boundaryPoint.intensity > middleCurvature)
//            {
//                startIndex = middleIndex + 1;
//            }
//            else
//            {
//                targetIndex = middleIndex;
//                break;
//            }
//        }

//        if (boundaryPoint.intensity > boundary_->points[targetIndex].intensity)
//        {
//            boundary_->points.insert(boundary_->points.begin() + targetIndex + 1, boundaryPoint);
//        }
//        else
//        {
//            boundary_->points.insert(boundary_->points.begin() + targetIndex, boundaryPoint);
//        }
//    }
//}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::joinSortedPoints()
{
    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(boundary_);

    for (int i = boundary_->points.size() - 1; i >= 0; i--)
    {
        if (boundary_->points[i].intensity != 0)
            continue;

        std::vector<int> segment;
        segment.push_back(i);
        boundary_->points[i].intensity = segments_.size() + 1;

        Eigen::Vector3f lastVector;
        int pointIndex = i;
        int stackedPointIndex = -1;
        while (true)
        {
            std::vector<int> neighbourIndices;
            std::vector<float> neighbourDistants;
//            tree.nearestKSearch(pointIndex, segment_k_search_, neighbourIndices, neighbourDistants);
            tree.radiusSearch(pointIndex, segment_distance_threshold_, neighbourIndices, neighbourDistants);

            std::vector<int> availableNeighbourIndices;
            for (int ni = 1; ni < neighbourIndices.size(); ni++)
            {
                pcl::PointXYZI nP = boundary_->points[neighbourIndices[ni]];
                if (nP.intensity != 0)
                {
                    continue;
                }
                if (neighbourDistants[ni] > segment_distance_threshold_)
                {
                    continue;
                }
                availableNeighbourIndices.push_back(neighbourIndices[ni]);
            }

            if (availableNeighbourIndices.empty())
            {
                break;
            }

            pcl::PointXYZI p0 = boundary_->points[pointIndex];
            pcl::PointXYZI p1 = boundary_->points[availableNeighbourIndices[0]];

            pcl::Vector3fMap ep0 = p0.getVector3fMap();
            pcl::Vector3fMap ep1 = p1.getVector3fMap();
            Eigen::Vector3f v1 = ep1 - ep0;
            v1.normalize();

            double maxAngle = 0;
            double minAngle = M_PI;
            int maxAngleIndex = -1;
            int minAngleIndex = -1;
            for (int ai = 0; ai < availableNeighbourIndices.size(); ai++)
            {
                pcl::PointXYZI pn = boundary_->points[availableNeighbourIndices[ai]];
                pcl::Vector3fMap epn = pn.getVector3fMap();
                Eigen::Vector3f vn = epn - ep0;
                vn.normalize();
                double angle = std::abs(std::acos(v1.dot(vn)));
                if (angle > maxAngle)
                {
                    maxAngle = angle;
                    maxAngleIndex = ai;
                }
                if (angle < minAngle)
                {
                    minAngle = angle;
                    minAngleIndex = ai;
                }
            }

            if (segment.size() == 1)
            {
                lastVector = v1;
                pointIndex = availableNeighbourIndices[0];
                segment.push_back(pointIndex);

                // 查找是否存在另一个方向，从有效近邻点中找一个与当前确定的增长方向角度最大的近邻点方向作为另一方向的延伸
                if (maxAngleIndex != -1)
                {
                    stackedPointIndex = availableNeighbourIndices[maxAngleIndex];
                    //std::cout << "Found another direction to search." << std::endl;
                }

                continue;
            }

            /*if (minAngleIndex >= 0)
                pointIndex = availableNeighbourIndices[minAngleIndex];
            else*/
            pointIndex = availableNeighbourIndices[0];
            boundary_->points[pointIndex].intensity = segments_.size() + 1;
            segment.push_back(pointIndex);
        }

        // 判断另一个方向
        lastVector.fill(0);
        if (stackedPointIndex >= 0)
        {
            pointIndex = stackedPointIndex;

            pcl::PointXYZI p0 = boundary_->points[i];
            pcl::PointXYZI p1 = boundary_->points[pointIndex];
            pcl::Vector3fMap ep0 = p0.getVector3fMap();
            pcl::Vector3fMap ep1 = p1.getVector3fMap();
            Eigen::Vector3f v1 = ep1 - ep0;
            v1.normalize();
            lastVector = v1;
            segment.insert(segment.begin(), pointIndex);
        }

        while (stackedPointIndex >= 0)
        {
            std::vector<int> neighbourIndices;
            std::vector<float> neighbourDistants;
//            tree.nearestKSearch(pointIndex, segment_k_search_, neighbourIndices, neighbourDistants);
            tree.radiusSearch(pointIndex, segment_distance_threshold_, neighbourIndices, neighbourDistants);

            std::vector<int> availableNeighbourIndices;
            for (int ni = 1; ni < neighbourIndices.size(); ni++)
            {
                pcl::PointXYZI nP = boundary_->points[neighbourIndices[ni]];
                if (nP.intensity != 0)
                {
                    continue;
                }
                if (neighbourDistants[ni] > segment_distance_threshold_)
                {
                    continue;
                }
                availableNeighbourIndices.push_back(neighbourIndices[ni]);
            }

            if (availableNeighbourIndices.empty())
            {
                break;
            }

            pcl::PointXYZI p0 = boundary_->points[pointIndex];
            pcl::PointXYZI p1 = boundary_->points[availableNeighbourIndices[0]];

            pcl::Vector3fMap ep0 = p0.getVector3fMap();
            pcl::Vector3fMap ep1 = p1.getVector3fMap();
            Eigen::Vector3f v1 = ep1 - ep0;
            v1.normalize();

            double maxAngle = 0;
            double minAngle = M_PI;
            int maxAngleIndex = -1;
            int minAngleIndex = -1;
            for (int ai = 0; ai < availableNeighbourIndices.size(); ai++)
            {
                pcl::PointXYZI pn = boundary_->points[availableNeighbourIndices[ai]];
                pcl::Vector3fMap epn = pn.getVector3fMap();
                Eigen::Vector3f vn = epn - ep0;
                vn.normalize();
                double angle = std::abs(std::acos(v1.dot(vn)));
                if (angle > maxAngle)
                {
                    maxAngle = angle;
                    maxAngleIndex = ai;
                }
                if (angle < minAngle)
                {
                    minAngle = angle;
                    minAngleIndex = ai;
                }
            }

            if (segment.size() == 1)
            {
                lastVector = v1;
                pointIndex = availableNeighbourIndices[0];
                segment.insert(segment.begin(), pointIndex);
                continue;
            }

            /*if (minAngleIndex >= 0)
                pointIndex = availableNeighbourIndices[minAngleIndex];
            else*/
            pointIndex = availableNeighbourIndices[0];
            boundary_->points[pointIndex].intensity = segments_.size() + 1;
            segment.insert(segment.begin(), pointIndex);
        }

        if (segment.size() > min_line_len_)
        {
            segments_.push_back(segment);
            //std::cout << std::setw(10) << "segment " << std::setw(4) << segments_.size() - 1 << ": " << segment.size() << "\t" << boundary_->points[i].x << ", " << boundary_->points[i].y << ", " << boundary_->points[i].z << std::endl;
            /*for (int j = 0; j < segment.size(); j++)
            {
                tmp2->push_back(boundary_->points[segment[j]]);
            }*/
        }
    }

    std::cout << "segment size: " << segments_.size() << std::endl;
    std::cout << "edge points size: " << boundary_->size() << std::endl;

    size_t maxLength = 0;
    int maxIndex = 0;
    for (int s = 0; s < segments_.size(); s++)
    {
        if (maxLength < segments_[s].size())
        {
            maxLength = segments_[s].size();
            maxIndex = s;
        }
    }
    qDebug().noquote().nospace() << "max segment length is " << maxLength << ", index is " << maxIndex;
//    std::vector<int> maxSegment = segments_[maxIndex];
//    segments_.clear();
//    segments_.push_back(maxSegment);

//    for (int s = 0; s < segments_.size(); s++)
//    {
//        if (segments_[s].size() < 1500)
//            continue;
//        std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
//        for (int i = 0; i < segments_[s].size(); i++)
//        {
//            double distance = 0;
//            if (i > 0)
//            {
//                double dx = boundary_->points[segments_[s][i]].x - boundary_->points[segments_[s][i - 1]].x;
//                double dy = boundary_->points[segments_[s][i]].y - boundary_->points[segments_[s][i - 1]].y;
//                double dz = boundary_->points[segments_[s][i]].z - boundary_->points[segments_[s][i - 1]].z;
//                distance = std::sqrt(dx * dx + dy * dy + dz * dz);
//            }
//            std::cout << "[" << std::setw(10) << boundary_->points[segments_[s][i]].x << ", " << std::setw(10) << boundary_->points[segments_[s][i]].y << ", " << std::setw(10) << boundary_->points[segments_[s][i]].z << "] \t" << std::setw(10) << distance << std::endl;
//        }
//    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::extractLinesFromSegment(const std::vector<int> &segment, int segmentNo)
{
    //int length = segments_[i].size();
    int index = 0;

    while (index < segment.size() - min_line_len_)
    {
        bool valid = false;
        Eigen::Vector3f dir;
        Eigen::Vector3f meanPoint;
        dir.fill(0);
        meanPoint.fill(0);

        while (index < segment.size() - min_line_len_)
        {
            float error = lineFit(segment, index, min_line_len_, dir, meanPoint);
            if (error <= pca_error_threshold_)
            {
                valid = true;
                break;
            }

            index++;
        }

        if (!valid)
            continue;

        int startIndex = index;
        int endIndex = startIndex + min_line_len_;

        while (endIndex < segment.size())
        {
            int currentIndex = endIndex;
            int badPointCount = 0;
            int goodPointCount = 0;

            while (endIndex < segment.size() && currentIndex < segment.size())
            {
                float distance = distanceToLine(boundary_->points[segment[currentIndex]], dir, meanPoint);
                if (distance <= pca_error_threshold_)
                {
                    endIndex++;
                    goodPointCount++;
                    badPointCount = 0;
                }
                else
                {
                    badPointCount++;
                    if (badPointCount >= 5)
                    {
                        break;
                    }
                }

                currentIndex++;
            }

            if (goodPointCount >= 2)
            {
                int len = endIndex - startIndex;
                lineFit(segment, startIndex, len, dir, meanPoint);
            }

            if (goodPointCount < 2 || endIndex >= segment.size())
            {
                if (endIndex >= segment.size())
                    endIndex = segment.size() - 1;

                //float sx, sy, ex, ey;
                int linePointIndex = startIndex;
                /*while (linePointIndex < segment.size() && distanceToLine(boundary_->points[segment[linePointIndex]], dir, meanPoint) > pca_error_threshold_)
                    linePointIndex++;*/
                startIndex = linePointIndex;
                Eigen::Vector3f startPoint = closedPointOnLine(boundary_->points[segment[startIndex]], dir, meanPoint);

                linePointIndex = endIndex;
                /*while (linePointIndex >= 0 && distanceToLine(boundary_->points[segment[linePointIndex]], dir, meanPoint) > pca_error_threshold_)
                    linePointIndex--;*/
                endIndex = linePointIndex;
                Eigen::Vector3f endPoint = closedPointOnLine(boundary_->points[segment[endIndex]], dir, meanPoint);

//                std::cout << "segment " << segmentNo << ": "
//                    << "["
//                    << boundary_->points[segment[startIndex]].x << ", "
//                    << boundary_->points[segment[startIndex]].y << ", "
//                    << boundary_->points[segment[startIndex]].z << "] "
//                    << startPoint.transpose() << ",\t"
//                    << "["
//                    << boundary_->points[segment[endIndex]].x << ", "
//                    << boundary_->points[segment[endIndex]].y << ", "
//                    << boundary_->points[segment[endIndex]].z << "] "
//                    << endPoint.transpose() << ", " << (endPoint - startPoint).norm() << std::endl;
                LineSegment line(startPoint, endPoint, segmentNo);
                unifyLineDirection(line);
                lines_.push_back(line);

                index = endIndex + 1;
                break;
            }
        }
    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::mergeCollinearLines()
{
    if (lines_.empty())
        return;

    std::vector<LineSegment> src = lines_;
    std::vector<LineSegment> dst;

    std::vector<LineSegment>* lines1 = &src;
    std::vector<LineSegment>* lines2 = &dst;

    int srcSize = static_cast<int>(lines1->size());
    int dstSize = 0;
    while (srcSize != dstSize)
    {
        lines2->clear();
        lines2->push_back((*lines1)[0]);

        for (int i = 1; i < lines1->size(); i++)
        {
            LineSegment currentLine = (*lines1)[i];
            LineSegment lastMergedLine = lines2->back();

            // 进入了下一个segment
            if (currentLine.segmentNo() != lastMergedLine.segmentNo())
            {
                lines2->push_back(currentLine);
                continue;
            }

            LINE_ORDER order;
            float lineDistance = linesDistance(lastMergedLine, currentLine, order);
            if (lineDistance >= max_distance_between_two_lines_)    // 两条线之间的距离太大，不考虑
            {
                lines2->push_back(currentLine);
                continue;
            }

            if (!isLinesCollinear2(lastMergedLine, currentLine)) // 不共线就直接下一条
            {
                lines2->push_back(currentLine);
                continue;
            }

            switch (order)
            {
            case LO_SS:
                lastMergedLine.setStart(currentLine.end());
                break;
            case LO_SE:
                lastMergedLine.setStart(currentLine.start());
                break;
            case LO_ES:
                lastMergedLine.setEnd(currentLine.end());
                break;
            case LO_EE:
                lastMergedLine.setEnd(currentLine.start());
                break;
            }
            (*lines2)[lines2->size() - 1] = lastMergedLine;
        }

        srcSize = lines1->size();
        dstSize = lines2->size();

        std::vector<LineSegment>* tmp = lines1;
        lines1 = lines2;
        lines2 = tmp;
    }

    mergedLines_ = dst;
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::mergeCollinearLines2()
{
    if (lines_.empty())
        return;

    std::vector<LineSegment> src = lines_;
    std::vector<LineSegment> dst;

    //std::vector<LineSegment>* pSrc = &src;
    //std::vector<LineSegment>* pDst = &dst;

    //int srcSize = pSrc->size();
    int lastSize = -1;
    while (lastSize != dst.size())
    {
        lastSize = dst.size();
        dst.clear();

        while (!src.empty())
        {
            LineSegment currentLine = src.back();
            src.pop_back();

            int index = 0;
            while (index < src.size())
            {
                LineSegment candidateLine = src[index];

                LINE_ORDER order;
                float lineDistance = linesDistance(currentLine, candidateLine, order);
                if (lineDistance >= max_distance_between_two_lines_)    // 两条线之间的距离太大，不考虑
                {
                    index++;
                    continue;
                }

                if (!isLinesCollinear(currentLine, candidateLine)) // 不共线就直接下一条
                {
                    index++;
                    continue;
                }

                switch (order)
                {
                case LO_SS:
                    currentLine.setStart(candidateLine.end());
                    break;
                case LO_SE:
                    currentLine.setStart(candidateLine.start());
                    break;
                case LO_ES:
                    currentLine.setEnd(candidateLine.end());
                    break;
                case LO_EE:
                    currentLine.setEnd(candidateLine.start());
                    break;
                }

                src.erase(src.begin() + index);
            }
            //(*pDst)[pDst->size() - 1] = lastMergedLine;

            dst.push_back(currentLine);
        }
        src = dst;
    }

    mergedLines_ = dst;
}

template<typename PointInT, typename PointOutT>
bool LineExtractor<PointInT, PointOutT>::isLinesCollinear(const LineSegment &line1, const LineSegment &line2)
{
    const LineSegment* longer = &line1;
    const LineSegment* shorter = &line2;

    if (line1.length() < line2.length())
    {
        longer = &line2;
        shorter = &line1;
    }

    float distance = distanceToLine(shorter->start(), longer->direction(), longer->middle());
    distance += distanceToLine(shorter->middle(), longer->direction(), longer->middle());
    distance += distanceToLine(shorter->end(), longer->direction(), longer->middle());

    distance /= 3.0f;

    if (distance >= max_error_)
        return false;

    return true;
}

template<typename PointInT, typename PointOutT>
bool LineExtractor<PointInT, PointOutT>::isLinesCollinear2(const LineSegment &line1, const LineSegment &line2)
{
    double angle = std::abs(std::acos(line1.direction().normalized().dot(line2.direction().normalized())));
    //std::cout << angle << ", " << line1.length() << ", " << line2.length() << std::endl;

    if (angle <= 90 && angle >= max_angle_error_)
        return false;
    else if (angle > 90 && (M_PI - angle) >= max_angle_error_)
        return false;

    return true;
}

template<typename PointInT, typename PointOutT>
float LineExtractor<PointInT, PointOutT>::linesDistance(const LineSegment &line1, const LineSegment &line2, LineExtractor::LINE_ORDER &order)
{
    // 计算头之间的距离
    float distance = (line1.start() - line2.start()).norm();
    float minDistance = distance;
    order = LO_SS;

    // 计算头尾之间的距离
    distance = (line1.start() - line2.end()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_SE;
    }

    // 计算尾头之间的距离
    distance = (line1.end() - line2.start()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_ES;
    }

    // 计算尾尾之间的距离
    distance = (line1.end() - line2.end()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_EE;
    }

    return minDistance;
}

template<typename PointInT, typename PointOutT>
float LineExtractor<PointInT, PointOutT>::lineFit(const std::vector<int> &segment, int index, int length, Eigen::Vector3f &dir, Eigen::Vector3f &meanPoint)
{
    std::vector<int> indices;
    for (int i = index; i < index + length; i++)
    {
        indices.push_back(segment[i]);
        //std::cout << std::setw(10) << cloud->points[segment[i]].x << ", " << std::setw(10) << cloud->points[segment[i]].y << ", " << std::setw(10) << cloud->points[segment[i]].z << std::endl;
    }

    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(boundary_);
    pca.setIndices(pcl::IndicesPtr(new std::vector<int>(indices)));
    Eigen::Vector3f eigenValue = pca.getEigenVectors().col(0).normalized();
    Eigen::Vector4f mean4f = pca.getMean();
    meanPoint = mean4f.head(3);
    dir = eigenValue;
    //std::cout << eigenValue.transpose() << ", " << eigenValue.normalized().transpose() << ", " << mean4f.transpose() << ", " << meanPoint.transpose() << std::endl;

    float sumDistance = 0;
    for (int i = index; i < index + length; i++)
    {
        float distance = distanceToLine(boundary_->points[segment[i]], dir, meanPoint);
        sumDistance += distance;
        //std::cout << distance << std::endl;
    }
    float error = sumDistance / length;
    //std::cout << "mean distance: " << error << std::endl;
    return error;
}

template<typename PointInT, typename PointOutT>
float LineExtractor<PointInT, PointOutT>::distanceToLine(pcl::PointXYZI &point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint)
{
    pcl::Vector3fMap ep = point.getVector3fMap();
    return distanceToLine(ep, dir, meanPoint);
}

template<typename PointInT, typename PointOutT>
float LineExtractor<PointInT, PointOutT>::distanceToLine(Eigen::Vector3f point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint)
{
    Eigen::Vector3f ev = point - meanPoint;
    Eigen::Vector3f epv = ev - dir * (ev.dot(dir));
    float distance = epv.norm();
    return distance;
}

template<typename PointInT, typename PointOutT>
Eigen::Vector3f LineExtractor<PointInT, PointOutT>::closedPointOnLine(pcl::PointXYZI &point, Eigen::Vector3f &dir, Eigen::Vector3f meanPoint)
{
    pcl::Vector3fMap ep = point.getVector3fMap();
    Eigen::Vector3f ev = ep - meanPoint;
    Eigen::Vector3f closedPoint = meanPoint + dir * (ev.dot(dir));
    return closedPoint;
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::generateLineCloud()
{
    for (std::vector<LineSegment>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
    {
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = i->start();
        middle.getVector3fMap() = i->middle();
        end.getVector3fMap() = i->end();
        lineCloud_->push_back(start);
        lineCloud_->push_back(middle);
        lineCloud_->push_back(end);
    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::unifyLineDirection(LineSegment &line)
{
    // 统一线段方向
    Eigen::Vector3f dir = line.direction().normalized();
    float xProj = std::abs(dir.x());
    float yProj = std::abs(dir.y());
    float zProj = std::abs(dir.z());

    //std::cout << xProj << ", " << yProj << ", " << zProj << std::endl;
    //std::cout << dir.eigenvalues() << std::endl;

    int maxAxis = 0;
    if (xProj > yProj)
    {
        if (xProj > zProj)
        {
            maxAxis = 0;
        }
        else if (zProj > yProj)
        {
            maxAxis = 2;
        }
    }
    else
    {
        if (yProj > zProj)
        {
            maxAxis = 1;
        }
        else
        {
            maxAxis = 2;
        }
    }

    bool inverse = false;
    if (maxAxis == 0 && dir.x() < 0)
        inverse = true;
    if (maxAxis == 1 && dir.y() < 0)
        inverse = true;
    if (maxAxis == 2 && dir.z() < 0)
        inverse = true;

    if (inverse)
    {
        line.reverse();
    }
    // 统一线段方向结束
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::linesSortingByLength(std::vector<LineSegment> &lines)
{
    std::sort(lines.begin(), lines.end(), LineCompare);
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::createLinesTree(const std::vector<LineSegment> &lines)
{
    if (lines.empty())
        return;

    mergedLines_.clear();
    int count = 0;
    for (std::vector<LineSegment>::const_iterator i = lines.begin(); i != lines.end(); i++)
    {
        LineTreeNode *node = new LineTreeNode(*i);
        if (root_ == nullptr)
        {
            root_ = node;
            mergedLines_.push_back(*i);
            continue;
        }

        qDebug() << count++ << "add node:" << node;
        addLineTreeNode(node);
    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::extracLinesClusters()
{
    LineTreeNode *curr = root_;
//    LineTreeNode *prev = nullptr;

    LineCluster *cluster = nullptr;

    // move to chain leaf node
    while (curr->hasLeftChild())
    {
        curr = curr->leftChild();
    }

    int count = 0;
    while (curr)
    {
        qDebug() << "line address:" << curr;

        if (curr->accessed())
            break;

        if (!cluster)
            cluster = new LineCluster;
        cluster->addLine(curr->line());
        for (int i = 0; i < curr->sideLines().size(); i++)
        {
            cluster->addLine(curr->sideLines()[i]->line());
        }
        curr->setAccessed();
        count++;

        if (curr->isLeftChild())
        {
            if (curr->chainDistance() > lines_chain_distance_threshold_)
            {
                // 一个聚集搜索完毕
                lineClusters_.append(cluster);
                cluster = nullptr;
            }
            curr = curr->parent();
        }
        else if (curr->isRightRoot())
        {
            // 一个聚集搜索完毕
            lineClusters_.append(cluster);
            cluster = nullptr;

            if (curr->hasRightChild())
            {
                curr = curr->rightChild();
                // move to chain leaf node
                while (curr->hasLeftChild())
                {
                    curr = curr->leftChild();
                }
            }
            else
            {
                break;
            }
        }
    }
    qDebug() << "iterate count " << count;
    for (int i = 0; i < lineClusters_.size(); i++)
    {
        qDebug().noquote() << "cluster" << i << ": cluster size:" << lineClusters_[i]->size();
    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::addLineTreeNode(LineTreeNode *node)
{
    LineTreeNode *curr = root_;

    float distance = 0;
    float angle = 0;
    float chainDistance = 0;
    LINE_RELATIONSHIP lineRel = LR_NONE;
    while (curr)
    {
//        qDebug() << "add line. curr " << curr;
        LineSegment longLine = curr->line();
        LineSegment shortLine = node->line();
        compareLines(longLine, shortLine, distance, angle, chainDistance, lineRel);

        if (lineRel == LR_SIDE)
        {
            node->setDistance(distance);
            node->setChainDistance(chainDistance);
            curr->addSideChild(node);
            curr = nullptr;
        }
        else if (lineRel == LR_CHAIN_BW)
        {
            if (curr->hasLeftChild())
            {
                curr = curr->leftChild();
            }
            else
            {
                curr->addLeftChild(node);
                curr = nullptr;
            }
            node->setDistance(distance);
            node->setChainDistance(chainDistance);
        }
        else if (lineRel == LR_CHAIN_FW)
        {
            if (curr->hasParent())
            {
                if (curr->isLeftChild())
                {
                    curr->parent()->addLeftChild(node);
                }
                else if (curr->isRightChild())
                {
                    curr->parent()->addRightChild(node);
                }
            }
            else
            {
                root_ = node;
            }

            if (curr->hasRightChild())
            {
                node->addRightChild(curr->rightChild());
                curr->addRightChild(nullptr);
            }
            node->addLeftChild(curr);
            curr->setDistance(distance);
            curr->setChainDistance(chainDistance);
            curr = nullptr;
        }
        else
        {
            curr = findRightRoot(curr);
            if (curr->hasRightChild())
            {
                curr = curr->rightChild();
            }
            else
            {
                curr->addRightChild(node);
                curr = nullptr;
            }
            node->setDistance(distance);
            node->setChainDistance(chainDistance);
        }
    }
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::compareLines(LineSegment &longLine, LineSegment &shortLine, float &distance, float &angle, float &chainDistance, LINE_RELATIONSHIP &lineRel)
{
    distance = 0;
    angle = 0;
    chainDistance = 0;
    lineRel = LR_NONE;

    Eigen::Vector3f ptS1 = longLine.start();
    Eigen::Vector3f ptS2 = shortLine.start();
    Eigen::Vector3f ptE1 = longLine.end();
    Eigen::Vector3f ptE2 = shortLine.end();

    Eigen::Vector3f line1 = longLine.direction();
    Eigen::Vector3f line2 = shortLine.direction();
    Eigen::Vector3f line1Dir = line1.normalized();
    Eigen::Vector3f line2Dir = line2.normalized();

    float theta;
    bool sameDirection = shortLine.applyAnotherLineDirection(longLine, theta, lines_cluster_angle_threshold_);
    angle = theta;
    if (!sameDirection)
    {
        lineRel = LR_NONE;
        return;
    }

    Eigen::Vector3f lineS1S2 = ptS2 - ptS1;
    Eigen::Vector3f lineE1E2 = ptE2 - ptE1;

    Eigen::Vector3f pt1 = ptS1;
    Eigen::Vector3f pt2 = ptS2;
    Eigen::Vector3f lineEndPoints = lineS1S2;
    if (lineS1S2.isZero())
    {
        if (lineE1E2.isZero())
        {
            angle = 0;
            distance = 0;
            lineRel = LR_SIDE;
            qDebug().nospace().noquote() << "distance: " << distance << ", theta = " << qRadiansToDegrees(theta);
            return;
        }
        else
        {
            pt1 = ptE1;
            pt2 = ptE2;
            line1 = -line1;
            line2 = -line2;
            line1Dir = -line1Dir;
            line2Dir = -line2Dir;
            lineEndPoints = lineE1E2;
        }
    }

//    Eigen::Vector3f lineM1M2Dir = line1Dir.cross(line2Dir);
//    lineM1M2Dir.normalize();
//    if (lineM1M2Dir.isZero())
//    {
//        distance = 0;
//    }
//    distance = qAbs(lineEndPoints.dot(lineM1M2Dir));

    distance = longLine.averageDistance(shortLine);

//    qDebug().nospace().noquote() << "distance: " << distance;
    if (distance >= lines_distance_threshold_)
    {
        lineRel = LR_NONE;
        return;
    }

    Eigen::Vector3f ptS2ProjOnLine1 = ptS1 + line1Dir * lineS1S2.dot(line1Dir);
    Eigen::Vector3f ptE2ProjOnLine1 = ptE1 + line1Dir * lineE1E2.dot(line1Dir);
    Eigen::Vector3f lineLine2ProjOnLine1 = ptE2ProjOnLine1 - ptS2ProjOnLine1;
    float lengthLine2ProjOnLine1 = lineLine2ProjOnLine1.norm();
    float lengthLongLine = longLine.length();
    float lengthUnionS1E2 = (ptE2ProjOnLine1 - ptS1).norm();
    float lengthUnionS2E1 = (ptS2ProjOnLine1 - ptE1).norm();
    float lengthUnion = lengthUnionS1E2 > lengthUnionS2E1 ? lengthUnionS1E2 : lengthUnionS2E1;

    if (lengthUnion <= (lengthLongLine + lengthLine2ProjOnLine1))
    {
        lineRel = LR_SIDE;
    }
    else
    {
        if (lengthUnionS1E2 > lengthUnionS2E1)
        {
            lineRel = LR_CHAIN_FW;
        }
        else
        {
            lineRel = LR_CHAIN_BW;
        }
        chainDistance = lengthUnion - (lengthLongLine + lengthLine2ProjOnLine1);
    }
    qDebug().nospace().noquote() << "theta = " << qRadiansToDegrees(theta) << ", chain distance = " << chainDistance << ", order = " << lineRel;

//    float sinTheta = qSin(theta);

//    Eigen::Vector3f lineM1M2 = lineM1M2Dir * distance;
//    Eigen::Vector3f pt2B = pt2 - lineM1M2;
//    Eigen::Vector3f line12B = pt2B - pt1;
//    Eigen::Vector3f line12BDir = line12B.normalized();
//    float lengthLine12B = line12B.norm();
//    float cosAlpha = line12BDir.dot(line1Dir);
//    float alpha = qAcos(cosAlpha);
//    float beta = M_PI - theta - alpha;
//    float length1M1 = qSin(beta) * lengthLine12B / sinTheta;
//    float length2M2 = qSin(alpha) * lengthLine12B / sinTheta;
//    Eigen::Vector3f ptM1 = pt1 + line1Dir * length1M1;
//    Eigen::Vector3f ptM2 = pt2 + line2Dir * length2M2;

//    qDebug().nospace().noquote() << "distance: " << distance << ", alpha = " << qRadiansToDegrees(alpha) << ", beta = " << qRadiansToDegrees(beta) << ", theta = " << qRadiansToDegrees(theta);
//    qDebug().nospace().noquote() << "  line1 length: " << line1.norm() << ", " << (ptM1 - ptS1).norm() << ", " << (ptM1 - ptE1).norm();
//    qDebug().nospace().noquote() << "  line2 length: " << line2.norm() << ", " << (ptM2 - ptS2).norm() << ", " << (ptM2 - ptE2).norm();
}

template<typename PointInT, typename PointOutT>
void LineExtractor<PointInT, PointOutT>::fetchSideNodes(LineTreeNode *node, std::vector<LineSegment> &cluster)
{
    QQueue<LineTreeNode *> nodes;

    if (!node)
        return;

    nodes.enqueue(node);
    while (!nodes.empty())
    {
        LineTreeNode *node = nodes.dequeue();
        cluster.push_back(node->line());

        if (node->hasLeftChild())
        {
            nodes.enqueue(node->leftChild());
        }

        if (node->hasRightChild())
        {
            nodes.enqueue(node->rightChild());
        }
    }
}

template<typename PointInT, typename PointOutT>
LineTreeNode *LineExtractor<PointInT, PointOutT>::findRightRoot(LineTreeNode *node)
{
    if (node == nullptr)
        return nullptr;

    LineTreeNode *curr = node;
    while (!curr->isRightRoot())
    {
        curr = curr->parent();
    }
    return curr;
}

template<typename PointInT, typename PointOutT>
LineTreeNode *LineExtractor<PointInT, PointOutT>::findLeftLeaf(LineTreeNode *node)
{
    if (node == nullptr)
        return nullptr;

    LineTreeNode *curr = node;
    while (curr->hasLeftChild())
    {
        curr = curr->leftChild();
    }
    return curr;
}
