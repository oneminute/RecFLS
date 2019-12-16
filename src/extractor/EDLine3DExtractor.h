#ifndef EDLINE3DEXTRACTOR_H
#define EDLINE3DEXTRACTOR_H

#include <pcl/features/eigen.h>
#include <pcl/features/feature.h>
#include <pcl/features/pfh.h>

namespace pcl
{
    /*struct EDLine3DSimpleDescriptor
    {
        float a;
        float b;
        float c;
        float d;
    };*/

    struct EDLine3D
    {
        EDLine3D(pcl::PointXYZI _start, pcl::PointXYZI _end, int _segmentNo) :
            start(_start),
            end(_end),
            segmentNo(_segmentNo)
        {}

        EDLine3D(Eigen::Vector3f _start, Eigen::Vector3f _end, int _segmentNo) :
            segmentNo(_segmentNo)
        {
            start.x = _start.x();
            start.y = _start.y();
            start.z = _start.z();
            end.x = _end.x();
            end.y = _end.y();
            end.z = _end.z();
        }

        Eigen::Vector3f eStart() const
        {
            return start.getVector3fMap();
        }

        Eigen::Vector3f eEnd() const
        {
            return end.getVector3fMap();
        }

        Eigen::Vector3f eMiddle() const
        {
            return (eStart() + eEnd()) / 2;
        }

        pcl::PointXYZI middle() const
        {
            pcl::PointXYZI mPoint;
            mPoint.x = (start.x + end.x) / 2;
            mPoint.y = (start.y + end.y) / 2;
            mPoint.z = (start.z + end.z) / 2;
            mPoint.intensity = start.intensity;
            return mPoint;
        }

        float length() const
        {
            return (eEnd() - eStart()).norm();
        }

        Eigen::Vector3f direction() const
        {
            return eEnd() - eStart();
        }

        void generateShotDescriptor(float minLength, float maxLength, pcl::Vector3fMap minPoint, pcl::Vector3fMap maxPoint)
        {
            Eigen::Vector3f s = eStart() - minPoint;
            Eigen::Vector3f m = eMiddle() - minPoint;
            Eigen::Vector3f e = eEnd() - minPoint;

            Eigen::Vector3f delta = maxPoint - minPoint;

            shotDescriptor[0] = s.x() / delta.x();
            shotDescriptor[1] = s.y() / delta.y();
            shotDescriptor[2] = s.z() / delta.z();
            shotDescriptor[3] = m.x() / delta.x();
            shotDescriptor[4] = m.y() / delta.y();
            shotDescriptor[5] = m.z() / delta.z();
            shotDescriptor[6] = e.x() / delta.x();
            shotDescriptor[7] = e.y() / delta.y();
            shotDescriptor[8] = e.z() / delta.z();

            Eigen::Vector3f dir = direction().normalized();
            shotDescriptor[9] = dir[0];
            shotDescriptor[10] = dir[1];
            shotDescriptor[11] = dir[2];
            shotDescriptor[12] = (length() - minLength) / (maxLength - minLength);
            shotDescriptor.normalize();
        }

        pcl::PointXYZI start;
        pcl::PointXYZI end;
        int segmentNo;

        Eigen::Matrix<float, 13, 1> shotDescriptor;
        Eigen::VectorXf longDescriptor;
    };

    template<typename PointInT = pcl::PointXYZ>
    struct PointCompare
    {
        bool operator() (const PointInT& v1, const PointInT& v2)
        {
            if (v1.x < v2.x)
            {
                return true;
            }
            else if (v1.x == v2.x)
            {
                if (v1.y < v2.y)
                {
                    return true;
                }
                else if (v1.y == v2.y)
                {
                    if (v1.z < v2.z)
                        return true;
                    else
                        return false;
                }
            }
            return false;
        }
    };

    enum LINE_ORDER
    {
        LO_SS = 0,
        LO_SE,
        LO_ES,
        LO_EE
    };

    template <typename PointInT, typename PointNT, typename PointOutT>
    class EDLine3DExtractor : public FeatureFromNormals<PointInT, PointNT, PointOutT>
    {
    public:
        typedef boost::shared_ptr<EDLine3DExtractor<PointInT, PointNT, PointOutT> > Ptr;
        typedef boost::shared_ptr<const EDLine3DExtractor<PointInT, PointNT, PointOutT> > ConstPtr;

        using Feature<PointInT, PointOutT>::feature_name_;
        using Feature<PointInT, PointOutT>::getClassName;
        using Feature<PointInT, PointOutT>::input_;
        using Feature<PointInT, PointOutT>::indices_;
        using Feature<PointInT, PointOutT>::k_;
        using Feature<PointInT, PointOutT>::tree_;
        using Feature<PointInT, PointOutT>::search_radius_;
        using Feature<PointInT, PointOutT>::search_parameter_;
        using Feature<PointInT, PointOutT>::surface_;
        using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;

        typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;

    public:
        /** \brief Empty constructor.
          * The angular threshold \a angle_threshold_ is set to M_PI / 2.0
          */
        EDLine3DExtractor() :
            angle_threshold_(static_cast<float> (M_PI) / 2.0f),
            segment_angle_threshold_((M_PI) / 4.0),
            segment_distance_threshold_(0.05),
            min_line_len_(9),
            pca_error_threshold_(0.008f),
            max_distance_between_two_lines_(0.05f),
            max_error_(0.05f),
            max_angle_error_(M_PI / 18),
            nr_subdiv_(5),
            pfh_tuple_(),
            d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI))), 
            lineCloud_(new pcl::PointCloud<pcl::PointXYZI>())
        {
            feature_name_ = "EDLine3DExtractor";
        };

        /** \brief Check whether a point is a boundary_ point in a planar patch of projected points given by indices.
           * \note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
           * \param[in] cloud a pointer to the input point cloud
           * \param[in] q_idx the index of the query point in \a cloud
           * \param[in] indices the estimated point neighbors of the query point
           * \param[in] u the u direction
           * \param[in] v the v direction
           * \param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
           */
        bool
            isBoundaryPoint(const pcl::PointCloud<PointInT> &cloud,
                int q_idx, const std::vector<int> &indices,
                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);

        /** \brief Check whether a point is a boundary_ point in a planar patch of projected points given by indices.
          * \note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
          * \param[in] cloud a pointer to the input point cloud
          * \param[in] q_point a pointer to the querry point
          * \param[in] indices the estimated point neighbors of the query point
          * \param[in] u the u direction
          * \param[in] v the v direction
          * \param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
          */
        bool
            isBoundaryPoint(const pcl::PointCloud<PointInT> &cloud,
                const PointInT &q_point,
                const std::vector<int> &indices,
                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);

        /** \brief Set the decision boundary_ (angle threshold) that marks points as boundary_ or regular.
          * (default \f$\pi / 2.0\f$)
          * \param[in] angle the angle threshold
          */
        inline void
            setAngleThreshold(float angle)
        {
            angle_threshold_ = angle;
        }

        /** \brief Get the decision boundary_ (angle threshold) as set by the user. */
        inline float
            getAngleThreshold()
        {
            return (angle_threshold_);
        }

        /** \brief Get a u-v-n coordinate system that lies on a plane defined by its normal
          * \param[in] p_coeff the plane coefficients (containing the plane normal)
          * \param[out] u the resultant u direction
          * \param[out] v the resultant v direction
          */
        inline void
            getCoordinateSystemOnPlane(const PointNT &p_coeff,
                Eigen::Vector4f &u, Eigen::Vector4f &v)
        {
            pcl::Vector4fMapConst p_coeff_v = p_coeff.getNormalVector4fMap();
            v = p_coeff_v.unitOrthogonal();
            u = p_coeff_v.cross3(v);
        }

        std::map<int, int> linesCompare(const std::vector<EDLine3D>& srcLines);

        std::map<int, int> linesCompare2(const std::vector<EDLine3D>& srcLines);

        std::map<int, int> linesCompare3(const std::vector<EDLine3D>& srcLines);

        std::vector<std::vector<int>> segments()
        {
            return segments_;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr getBoundary()
        {
            return boundary_;
        }

        std::vector<EDLine3D> getLines()
        {
            return lines_;
        }

        std::vector<EDLine3D> getMergedLines()
        {
            return mergedLines_;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr getLineCloud()
        {
            return lineCloud_;
        }

    protected:
        /** \brief Estimate whether a set of points is lying on surface boundaries using an angle criterion for all points
          * given in <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
          * setSearchMethod ()
          * \param[out] output the resultant point cloud model dataset that contains boundary_ point estimates
          */
        void computeFeature(PointCloudOut &output);

        void computeDescriptorFeature(EDLine3D& line);

        void computePointPFHSignature (const std::vector<int> &indices, int nr_split, Eigen::VectorXf &pfh_histogram);

        void addPointToSortedIndices(int index);

        void joinSortedPoints();

        void extractLinesFromSegment(const std::vector<int>& segment, int segmentNo);

        void mergeCollinearLines();

        void mergeCollinearLines2();

        bool isLinesCollinear(EDLine3D line1, EDLine3D line2);

        bool isLinesCollinear2(EDLine3D line1, EDLine3D line2);

        float linesDistance(EDLine3D& line1, EDLine3D& line2, LINE_ORDER& order);

        float lineFit(const std::vector<int>& segment, int index, int length, Eigen::Vector3f& dir, Eigen::Vector3f& meanPoint);

        float distanceToLine(pcl::PointXYZI& point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

        float distanceToLine(Eigen::Vector3f point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

        Eigen::Vector3f closedPointOnLine(pcl::PointXYZI& point, Eigen::Vector3f& dir, Eigen::Vector3f meanPoint);

        void generateLineCloud();

        /** \brief The decision boundary_ (angle threshold) that marks points as boundary_ or regular. (default \f$\pi / 2.0\f$) */
        float angle_threshold_;

        std::vector<std::vector<int>> segments_;

        double segment_angle_threshold_;
        double segment_distance_threshold_;
        int min_line_len_;
        float pca_error_threshold_;
        float max_distance_between_two_lines_;
        float max_error_;
        float max_angle_error_;
        
        int nr_subdiv_;
        Eigen::Vector4f pfh_tuple_;
        int f_index_[3];
        float d_pi_;

        //pcl::PointCloud<PointInT>::Ptr downloadSampledCloud_;
        pcl::PointCloud<pcl::PointXYZI>::Ptr boundary_;
        std::vector<EDLine3D> lines_;
        std::vector<EDLine3D> mergedLines_;

        std::map<pcl::PointXYZI, EDLine3D, PointCompare<pcl::PointXYZI>> pointsLineMap_;
        pcl::PointCloud<pcl::PointXYZI>::Ptr lineCloud_;
    };

    
}


#endif // EDLINE3DEXTRACTOR_H
