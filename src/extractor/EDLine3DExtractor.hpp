#include <pcl/features/boundary.h>
#include <pcl/geometry/triangle_mesh.h>
#include <pcl/common/pca.h>
#include <cfloat>
#include "EDLine3DExtractor.h"

bool EDLine3DCompare(const pcl::EDLine3D& l1, const pcl::EDLine3D& l2)
{
    return l1.length() < l2.length();
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> bool
pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::isBoundaryPoint(
    const pcl::PointCloud<PointInT> &cloud, int q_idx,
    const std::vector<int> &indices,
    const Eigen::Vector4f &u, const Eigen::Vector4f &v,
    const float angle_threshold)
{
    return (isBoundaryPoint(cloud, cloud.points[q_idx], indices, u, v, angle_threshold));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> bool
pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::isBoundaryPoint(
    const pcl::PointCloud<PointInT> &cloud, const PointInT &q_point,
    const std::vector<int> &indices,
    const Eigen::Vector4f &u, const Eigen::Vector4f &v,
    const float angle_threshold)
{
    if (indices.size() < 3)
        return (false);

    if (!pcl_isfinite(q_point.x) || !pcl_isfinite(q_point.y) || !pcl_isfinite(q_point.z))
        return (false);

    // Compute the angles between each neighboring point and the query point itself
    std::vector<float> angles(indices.size());
    float max_dif = FLT_MIN, dif;
    int cp = 0;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (!pcl_isfinite(cloud.points[indices[i]].x) ||
            !pcl_isfinite(cloud.points[indices[i]].y) ||
            !pcl_isfinite(cloud.points[indices[i]].z))
            continue;

        Eigen::Vector4f delta = cloud.points[indices[i]].getVector4fMap() - q_point.getVector4fMap();
        if (delta == Eigen::Vector4f::Zero())
            continue;

        angles[cp++] = atan2f(v.dot(delta), u.dot(delta)); // the angles are fine between -PI and PI too
    }
    if (cp == 0)
        return (false);

    angles.resize(cp);
    std::sort(angles.begin(), angles.end());

    // Compute the maximal angle difference between two consecutive angles
    for (size_t i = 0; i < angles.size() - 1; ++i)
    {
        dif = angles[i + 1] - angles[i];
        if (max_dif < dif)
            max_dif = dif;
    }
    // Get the angle difference between the last and the first
    dif = 2 * static_cast<float> (M_PI) - angles[angles.size() - 1] + angles[0];
    if (max_dif < dif)
        max_dif = dif;

    // Check results
    if (max_dif > angle_threshold)
        return (true);
    else
        return (false);
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline std::map<int, int> pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::linesCompare(const std::vector<EDLine3D>& srcLines)
{
    std::cout << lineCloud_->size() << std::endl;
    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(lineCloud_);
    float radius = 0.05;

    std::map<int, int> pairs;

    for (int i = 0; i < srcLines.size(); i++)
    {
        const EDLine3D& line = srcLines[i];
        std::vector<int> indices;
        std::vector<float> distances;
        std::vector<int> indicesResult;
        std::vector<float> distancesResult;

        tree.radiusSearch(line.start, radius, indicesResult, distancesResult);
        if (indicesResult.size() > 0)
            indices.insert(indices.end(), indicesResult.begin(), indicesResult.end());

        tree.radiusSearch(line.end, radius, indicesResult, distancesResult);
        if (indicesResult.size() > 0)
            indices.insert(indices.end(), indicesResult.begin(), indicesResult.end());

        tree.radiusSearch(line.middle(), radius, indicesResult, distancesResult);
        if (indicesResult.size() > 0)
            indices.insert(indices.end(), indicesResult.begin(), indicesResult.end());

        std::vector<int> lineIndices;
        for (int j = 0; j < indices.size(); j++)
        {
            int lineIndex = lineCloud_->points[indices[j]].intensity;
            //std::cout << "src line: " << i << ", candidate line: " << lineIndex << std::endl;
            if (std::find(lineIndices.begin(), lineIndices.end(), lineIndex) == lineIndices.end())
            {
                lineIndices.push_back(lineIndex);
                //std::cout << "src line: " << i << " with length: " << srcLines[i].length() << ", candidate line: " << lineIndex << " with length: " << mergedLines_[lineIndex].length() << std::endl;
            }
        }
        int targetLineIndex = -1;
        if (!lineIndices.empty())
        {
            Eigen::Matrix<float, Eigen::Dynamic, 4> candidatesMat;
            candidatesMat.resize(lineIndices.size(), 4);
            for (int j = 1; j < lineIndices.size(); j++)
            {
                //std::cout << mergedLines_[lineIndices[j]].direction() << std::endl;
                candidatesMat.row(j) = mergedLines_[lineIndices[j]].getSimpleDescriptor();
            }

            Eigen::MatrixXf result = candidatesMat * line.shotDescriptor();
            //std::cout << result << std::endl;

            float max = 0;
            int maxIndex = -1;
            for (int j = 0; j < result.rows(); j++)
            {
                if (max < result.col(0)[j])
                {
                    max = result.col(0)[j];
                    maxIndex = j;
                }
            }

            if (maxIndex >= 0 && max >= 0.95f)
            {
                targetLineIndex = lineIndices[maxIndex];
            }
            std::cout << i << " --> max: " << maxIndex << ", " << max << ", " << targetLineIndex << std::endl;
        }
        pairs.insert(std::make_pair(i, targetLineIndex));
    }
    return pairs;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline std::map<int, int> pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::linesCompare2(const std::vector<EDLine3D>& srcLines)
{
    Eigen::MatrixXf mDesc1;
    Eigen::MatrixXf mDesc2;

    mDesc1.resize(srcLines.size(), 13);
    mDesc2.resize(mergedLines_.size(), 13);

    for (int i = 0; i < srcLines.size(); i++)
    {
        mDesc1.row(i) = srcLines[i].shotDescriptor;
    }

    for (int i = 0; i < mergedLines_.size(); i++)
    {
        mDesc2.row(i) = mergedLines_[i].shotDescriptor;
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
        candidateM.resize(candidateLineIndices.size(), srcLines[i].longDescriptor.size());
        for (int j = 0; j < candidateLineIndices.size(); j++)
        {
            candidateM.row(j) = mergedLines_[candidateLineIndices[j]].longDescriptor;
        }
        std::cout << i << "(rough) --> max: " << maxIndex << ", " << max << ", " << candidateLineIndices.size() << std::endl;

        if (candidateLineIndices.size() > 0)
        {
            Eigen::VectorXf longResult = srcLines[i].longDescriptor.transpose() * candidateM.transpose();
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

template<typename PointInT, typename PointNT, typename PointOutT>
inline std::map<int, int> pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::linesCompare3(const std::vector<EDLine3D>& srcLines)
{
    Eigen::MatrixXf mDesc1;
    Eigen::MatrixXf mDesc2;

    mDesc1.resize(srcLines.size(), nr_subdiv_ * nr_subdiv_ * nr_subdiv_ * 3);
    mDesc2.resize(mergedLines_.size(), nr_subdiv_ * nr_subdiv_ * nr_subdiv_ * 3);

    for (int i = 0; i < srcLines.size(); i++)
    {
        mDesc1.row(i) = srcLines[i].longDescriptor;
    }

    for (int i = 0; i < mergedLines_.size(); i++)
    {
        mDesc2.row(i) = mergedLines_[i].longDescriptor;
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

        if (max >= 0.97f)
        {
            pairs.insert(std::make_pair(i, maxIndex));
            std::cout << i << " --> max: " << maxIndex << ", " << max << std::endl;
        }
    }

    return pairs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> void
pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::computeFeature(PointCloudOut &output)
{
    // Allocate enough space to hold the results
    // \note This resize is irrelevant for a radiusSearch ().
    std::vector<int> nn_indices(k_);
    std::vector<float> nn_dists(k_);

    Eigen::Vector4f u = Eigen::Vector4f::Zero(), v = Eigen::Vector4f::Zero();
    boundary_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

    //output.width = output.height = 0;
    //output.points.clear();
    output.is_dense = true;
    // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
    if (input_->is_dense)
    {
        // Iterating over the entire index vector
        for (size_t idx = 0; idx < indices_->size(); ++idx)
        {
            if (this->searchForNeighbors((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
            {
                //output.points[idx].boundary_point = std::numeric_limits<uint8_t>::quiet_NaN();
                output.is_dense = false;
                continue;
            }

            // Obtain a coordinate system on the least-squares plane
            //v = normals_->points[(*indices_)[idx]].getNormalVector4fMap ().unitOrthogonal ();
            //u = normals_->points[(*indices_)[idx]].getNormalVector4fMap ().cross3 (v);
            getCoordinateSystemOnPlane(normals_->points[(*indices_)[idx]], u, v);

            // Estimate whether the point is lying on a boundary_ surface or not
            if (isBoundaryPoint(*surface_, input_->points[(*indices_)[idx]], nn_indices, u, v, angle_threshold_))
            {
                output.points[(*indices_)[idx]].boundary_point = 1;
                addPointToSortedIndices(idx);
            }
            else
            {
                output.points[(*indices_)[idx]].boundary_point = 0;
            }

        }
    }
    else
    {
        // Iterating over the entire index vector
        for (size_t idx = 0; idx < indices_->size(); ++idx)
        {
            if (!isFinite((*input_)[(*indices_)[idx]]) ||
                this->searchForNeighbors((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
            {
                //output.points[idx].boundary_point = std::numeric_limits<uint8_t>::quiet_NaN();
                output.is_dense = false;
                continue;
            }

            // Obtain a coordinate system on the least-squares plane
            //v = normals_->points[(*indices_)[idx]].getNormalVector4fMap ().unitOrthogonal ();
            //u = normals_->points[(*indices_)[idx]].getNormalVector4fMap ().cross3 (v);
            getCoordinateSystemOnPlane(normals_->points[(*indices_)[idx]], u, v);

            // Estimate whether the point is lying on a boundary_ surface or not
            if (isBoundaryPoint(*surface_, input_->points[(*indices_)[idx]], nn_indices, u, v, angle_threshold_))
            {
                output.points[(*indices_)[idx]].boundary_point = 1;
                addPointToSortedIndices(idx);
            }
            else
            {
                output.points[(*indices_)[idx]].boundary_point = 0;
            }
        }
    }

    for (int i = 0; i < boundary_->points.size(); i++)
    {
        boundary_->points[i].intensity = 0;
    }

    joinSortedPoints();

    for (int i = 0; i < segments_.size(); i++)
    {
        extractLinesFromSegment(segments_[i], i);
    }

    mergeCollinearLines();

    std::sort(mergedLines_.begin(), mergedLines_.end(), EDLine3DCompare);

    float minLength = mergedLines_.begin()->length();
    float maxLength = minLength;
    for (std::vector<EDLine3D>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
    {
        float length = i->length();
        if (minLength > length)
            minLength = length;
        if (maxLength < length)
            maxLength = length;
    }

    // 重新为每一个线段的端点的intensity设值，值为线段在集合中的索引下标
    int number = 0;
    std::vector<EDLine3D> tmpLines;
    for (std::vector<EDLine3D>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
    {
        if (i->length() > 0.05f)
        {
            i->segmentNo = number;

            // 统一线段方向
            Eigen::Vector3f dir = i->direction().normalized();
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
            if (maxAxis == 0 && i->start.x > i->end.x)
                inverse = true;
            if (maxAxis == 1 && i->start.y > i->end.y)
                inverse = true;
            if (maxAxis == 2 && i->start.z > i->end.z)
                inverse = true;

            if (inverse)
            {
                pcl::PointXYZI pTmp = i->start;
                i->start = i->end;
                i->end = pTmp;
            }
            // 统一线段方向结束

            i->start.intensity = i->end.intensity = number;
            //i->generateSimpleDescriptor(minLength, maxLength);
            tmpLines.push_back(*i);
            number++;
        }
    }
    mergedLines_ = tmpLines;

    generateLineCloud();

    pcl::PointXYZI minPoint, maxPoint;
    pcl::getMinMax3D<pcl::PointXYZI>(*lineCloud_, minPoint, maxPoint);
    pcl::Vector3fMap minValue = minPoint.getVector3fMap(), maxValue = maxPoint.getVector3fMap();
    for (std::vector<EDLine3D>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
    {
        i->generateShotDescriptor(minLength, maxLength, minValue, maxValue);
        computeDescriptorFeature(*i);
    }

    std::cout << "surface size: " << surface_->size() << std::endl;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::computeDescriptorFeature(EDLine3D& line)
{
    float radius = 0.03f;
    std::vector<int> indices;
    std::vector<float> distances;
    std::vector<int> indicesResult;
    std::vector<float> distancesResult;
    PointInT start, end, middle;
    start.x = line.start.x;
    start.y = line.start.y;
    start.z = line.start.z;
    end.x = line.end.x;
    end.y = line.end.y;
    end.z = line.end.z;
    middle.x = line.middle().x;
    middle.y = line.middle().y;
    middle.z = line.middle().z;
    Eigen::VectorXf pfh;
    line.longDescriptor.setZero(nr_subdiv_ * nr_subdiv_ * nr_subdiv_ * 3);

    tree_->radiusSearch(start, radius, indicesResult, distancesResult);
    pfh.setZero(nr_subdiv_ * nr_subdiv_ * nr_subdiv_);
    if (indicesResult.size() > 0)
    {
        computePointPFHSignature(indicesResult, nr_subdiv_, pfh);
        for (int i = 0; i < pfh.size(); i++)
        {
            line.longDescriptor[i] = pfh[i];
        }     
    }
    else
    {
        for (int i = 0; i < pfh.size() * 3; i++)
        {
            line.longDescriptor[i] = std::numeric_limits<float>::quiet_NaN();
        }
        std::cout << "line " << line.segmentNo << " cannot generate long descriptor." << std::endl;
        return;
    }
    tree_->radiusSearch(middle, radius, indicesResult, distancesResult);
    if (indicesResult.size() > 0)
    {
        computePointPFHSignature(indicesResult, nr_subdiv_, pfh);
        for (int i = 0; i < pfh.size(); i++)
        {
            line.longDescriptor[pfh.size() + i] = pfh[i];
        }     
    }
    else
    {
        for (int i = 0; i < pfh.size() * 3; i++)
        {
            line.longDescriptor[i] = std::numeric_limits<float>::quiet_NaN();
        }
        return;
    }
    tree_->radiusSearch(end, radius, indicesResult, distancesResult);
    if (indicesResult.size() > 0)
    {
        computePointPFHSignature(indicesResult, nr_subdiv_, pfh);
        for (int i = 0; i < pfh.size(); i++)
        {
            line.longDescriptor[pfh.size() * 2 + i] = pfh[i];
        }     
    }
    else
    {
        for (int i = 0; i < pfh.size() * 3; i++)
        {
            line.longDescriptor[i] = std::numeric_limits<float>::quiet_NaN();
        }
        return;
    }

    line.longDescriptor.normalize();
    //std::cout << "line " << line.segmentNo << ": " << line.longDescriptor.minCoeff() << ", " << line.longDescriptor.mean() << ", " << line.longDescriptor.maxCoeff() << std::endl;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::computePointPFHSignature(const std::vector<int>& indices, int nr_split, Eigen::VectorXf & pfh_histogram)
{
    int h_index, h_p;

    // Clear the resultant point histogram
    pfh_histogram.setZero();

    // Factorization constant
    float hist_incr = 100.0f / static_cast<float> (indices.size() * (indices.size() - 1) / 2);

    std::pair<int, int> key;
    bool key_found = false;

    // Iterate over all the points in the neighborhood
    for (size_t i_idx = 0; i_idx < indices.size(); ++i_idx)
    {
        for (size_t j_idx = 0; j_idx < i_idx; ++j_idx)
        {
            // If the 3D points are invalid, don't bother estimating, just continue
            if (!isFinite(surface_->points[indices[i_idx]]) || !isFinite(surface_->points[indices[j_idx]]))
                continue;

            if (!pcl::computePairFeatures(
                surface_->points[indices[i_idx]].getVector4fMap(),
                normals_->points[indices[i_idx]].getNormalVector4fMap(),
                surface_->points[indices[j_idx]].getVector4fMap(),
                normals_->points[indices[j_idx]].getNormalVector4fMap(),
                pfh_tuple_[0], pfh_tuple_[1], pfh_tuple_[2], pfh_tuple_[3]))
                continue;

            // Normalize the f1, f2, f3 features and push them in the histogram
            f_index_[0] = static_cast<int> (floor(nr_split * ((pfh_tuple_[0] + M_PI) * d_pi_)));
            if (f_index_[0] < 0)         f_index_[0] = 0;
            if (f_index_[0] >= nr_split) f_index_[0] = nr_split - 1;

            f_index_[1] = static_cast<int> (floor(nr_split * ((pfh_tuple_[1] + 1.0) * 0.5)));
            if (f_index_[1] < 0)         f_index_[1] = 0;
            if (f_index_[1] >= nr_split) f_index_[1] = nr_split - 1;

            f_index_[2] = static_cast<int> (floor(nr_split * ((pfh_tuple_[2] + 1.0) * 0.5)));
            if (f_index_[2] < 0)         f_index_[2] = 0;
            if (f_index_[2] >= nr_split) f_index_[2] = nr_split - 1;

            // Copy into the histogram
            h_index = 0;
            h_p = 1;
            for (int d = 0; d < 3; ++d)
            {
                h_index += h_p * f_index_[d];
                h_p *= nr_split;
            }
            pfh_histogram[h_index] += hist_incr;
        }
    }
}

template<typename PointInT, typename PointNT, typename PointOutT>
void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::addPointToSortedIndices(int index)
{
    PointNT currentNormal = normals_->points[(*indices_)[index]];
    PointInT currentPoint = input_->points[(*indices_)[index]];
    double currentCurvature = currentNormal.curvature;
    pcl::PointXYZI boundaryPoint;
    boundaryPoint.x = currentPoint.x;
    boundaryPoint.y = currentPoint.y;
    boundaryPoint.z = currentPoint.z;
    boundaryPoint.intensity = currentCurvature;
    if (boundary_->points.empty())
    {
        boundary_->points.push_back(boundaryPoint);
    }
    else
    {
        // 二分查找
        int startIndex = 0;
        int endIndex = boundary_->size() - 1;
        int targetIndex = -1;
        while (true)
        {
            int middleIndex = (startIndex + endIndex) / 2;
            int length = endIndex - startIndex + 1;
            if (length <= 1)
            {
                targetIndex = middleIndex;
                break;
            }
            pcl::PointXYZI middlePoint = boundary_->points[middleIndex];
            double middleCurvature = middlePoint.intensity;
            if (currentCurvature < middleCurvature)
            {
                endIndex = middleIndex - 1;
            }
            else if (currentCurvature > middleCurvature)
            {
                startIndex = middleIndex + 1;
            }
            else
            {
                targetIndex = middleIndex;
                break;
            }
        }

        if (boundaryPoint.intensity > boundary_->points[targetIndex].intensity)
        {
            boundary_->points.insert(boundary_->points.begin() + targetIndex + 1, boundaryPoint);
        }
        else
        {
            boundary_->points.insert(boundary_->points.begin() + targetIndex, boundaryPoint);
        }
    }
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::joinSortedPoints()
{
    pcl::search::KdTree<PointXYZI> tree;
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
            tree.nearestKSearch(pointIndex, 5, neighbourIndices, neighbourDistants);

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
            tree.nearestKSearch(pointIndex, 5, neighbourIndices, neighbourDistants);

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

    /*for (int s = 0; s < segments_.size(); s++)
    {
        if (segments_[s].size() < 1500)
            continue;
        std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
        for (int i = 0; i < segments_[s].size(); i++)
        {
            double distance = 0;
            if (i > 0)
            {
                double dx = boundary_->points[segments_[s][i]].x - boundary_->points[segments_[s][i - 1]].x;
                double dy = boundary_->points[segments_[s][i]].y - boundary_->points[segments_[s][i - 1]].y;
                double dz = boundary_->points[segments_[s][i]].z - boundary_->points[segments_[s][i - 1]].z;
                distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            std::cout << "[" << std::setw(10) << boundary_->points[segments_[s][i]].x << ", " << std::setw(10) << boundary_->points[segments_[s][i]].y << ", " << std::setw(10) << boundary_->points[segments_[s][i]].z << "] \t" << std::setw(10) << distance << std::endl;
        }
    }*/
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::extractLinesFromSegment(const std::vector<int>& segment, int segmentNo)
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

                /*std::cout << "segment " << segmentNo << ": "
                    << "["
                    << boundary_->points[segment[startIndex]].x << ", "
                    << boundary_->points[segment[startIndex]].y << ", "
                    << boundary_->points[segment[startIndex]].z << "] "
                    << startPoint.transpose() << ",\t"
                    << "["
                    << boundary_->points[segment[endIndex]].x << ", "
                    << boundary_->points[segment[endIndex]].y << ", "
                    << boundary_->points[segment[endIndex]].z << "] "
                    << endPoint.transpose() << ", " << (endPoint - startPoint).norm() << std::endl;*/
                EDLine3D line(startPoint, endPoint, segmentNo);
                lines_.push_back(line);

                index = endIndex + 1;
                break;
            }
        }
    }
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::mergeCollinearLines()
{
    if (lines_.empty())
        return;

    std::vector<EDLine3D> src = lines_;
    std::vector<EDLine3D> dst;

    std::vector<EDLine3D>* lines1 = &src;
    std::vector<EDLine3D>* lines2 = &dst;

    int srcSize = lines1->size();
    int dstSize = 0;
    while (srcSize != dstSize)
    {
        lines2->clear();
        lines2->push_back((*lines1)[0]);

        for (int i = 1; i < lines1->size(); i++)
        {
            EDLine3D currentLine = (*lines1)[i];
            EDLine3D lastMergedLine = lines2->back();

            // 进入了下一个segment
            if (currentLine.segmentNo != lastMergedLine.segmentNo)
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

            if (!isLinesCollinear(lastMergedLine, currentLine)) // 不共线就直接下一条
            {
                lines2->push_back(currentLine);
                continue;
            }

            switch (order)
            {
            case LO_SS:
                lastMergedLine.start = currentLine.end;
                break;
            case LO_SE:
                lastMergedLine.start = currentLine.start;
                break;
            case LO_ES:
                lastMergedLine.end = currentLine.end;
                break;
            case LO_EE:
                lastMergedLine.end = currentLine.start;
                break;
            }
            (*lines2)[lines2->size() - 1] = lastMergedLine;
        }

        srcSize = lines1->size();
        dstSize = lines2->size();

        std::vector<EDLine3D>* tmp = lines1;
        lines1 = lines2;
        lines2 = tmp;
    }

    mergedLines_ = dst;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::mergeCollinearLines2()
{
    if (lines_.empty())
        return;

    std::vector<EDLine3D> src = lines_;
    std::vector<EDLine3D> dst;

    //std::vector<EDLine3D>* pSrc = &src;
    //std::vector<EDLine3D>* pDst = &dst;

    //int srcSize = pSrc->size();
    int lastSize = -1;
    while (lastSize != dst.size())
    {
        lastSize = dst.size();
        dst.clear();

        while (!src.empty())
        {
            EDLine3D currentLine = src.back();
            src.pop_back();

            int index = 0;
            while (index < src.size())
            {
                EDLine3D candidateLine = src[index];

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
                    currentLine.start = candidateLine.end;
                    break;
                case LO_SE:
                    currentLine.start = candidateLine.start;
                    break;
                case LO_ES:
                    currentLine.end = candidateLine.end;
                    break;
                case LO_EE:
                    currentLine.end = candidateLine.start;
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

template<typename PointInT, typename PointNT, typename PointOutT>
inline bool pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::isLinesCollinear(EDLine3D line1, EDLine3D line2)
{
    EDLine3D& longer = line1;
    EDLine3D& shorter = line2;

    if (line1.length() < line2.length())
    {
        longer = line2;
        shorter = line1;
    }

    float distance = distanceToLine(shorter.eStart(), longer.direction(), longer.eMiddle());
    distance += distanceToLine(shorter.eMiddle(), longer.direction(), longer.eMiddle());
    distance += distanceToLine(shorter.eEnd(), longer.direction(), longer.eMiddle());

    distance /= 3.0f;

    if (distance >= max_error_)
        return false;

    return true;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline bool pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::isLinesCollinear2(EDLine3D line1, EDLine3D line2)
{
    double angle = std::abs(std::acos(line1.direction().normalized().dot(line2.direction().normalized())));
    //std::cout << angle << ", " << line1.length() << ", " << line2.length() << std::endl;

    if (angle <= 90 && angle >= max_angle_error_)
        return false;
    else if (angle > 90 && (M_PI - angle) >= max_angle_error_)
        return false;

    return true;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline float pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::linesDistance(EDLine3D & line1, EDLine3D & line2, LINE_ORDER & order)
{
    // 计算头之间的距离
    float distance = (line1.eStart() - line2.eStart()).norm();
    float minDistance = distance;
    order = LO_SS;

    // 计算头尾之间的距离
    distance = (line1.eStart() - line2.eEnd()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_SE;
    }

    // 计算尾头之间的距离
    distance = (line1.eEnd() - line2.eStart()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_ES;
    }

    // 计算尾尾之间的距离
    distance = (line1.eEnd() - line2.eEnd()).norm();
    if (distance < minDistance)
    {
        minDistance = distance;
        order = LO_EE;
    }

    return minDistance;
}

template<typename PointInT, typename PointNT, typename PointOutT>
float pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::lineFit(const std::vector<int>& segment, int index, int length, Eigen::Vector3f& dir, Eigen::Vector3f& meanPoint)
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

template<typename PointInT, typename PointNT, typename PointOutT>
inline float pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::distanceToLine(pcl::PointXYZI & point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint)
{
    pcl::Vector3fMap ep = point.getVector3fMap();
    return distanceToLine(ep, dir, meanPoint);
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline float pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::distanceToLine(Eigen::Vector3f point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint)
{
    Eigen::Vector3f ev = point - meanPoint;
    Eigen::Vector3f epv = ev - dir * (ev.dot(dir));
    float distance = epv.norm();
    return distance;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline Eigen::Vector3f pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::closedPointOnLine(pcl::PointXYZI & point, Eigen::Vector3f & dir, Eigen::Vector3f meanPoint)
{
    pcl::Vector3fMap ep = point.getVector3fMap();
    Eigen::Vector3f ev = ep - meanPoint;
    Eigen::Vector3f closedPoint = meanPoint + dir * (ev.dot(dir));
    return closedPoint;
}

template<typename PointInT, typename PointNT, typename PointOutT>
inline void pcl::EDLine3DExtractor<PointInT, PointNT, PointOutT>::generateLineCloud()
{
    for (std::vector<EDLine3D>::iterator i = mergedLines_.begin(); i != mergedLines_.end(); i++)
    {
        lineCloud_->push_back(i->start);
        lineCloud_->push_back(i->end);
        lineCloud_->push_back(i->middle());
    }

    lineCloud_->height = 1;
    lineCloud_->width = lineCloud_->points.size();
    lineCloud_->is_dense = false;
}

