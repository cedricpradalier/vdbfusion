#pragma once

#include "VDBVolume.h"


namespace vdbfusion {

class VDBColoredVolume : public VDBVolume {
public:
    VDBColoredVolume(float voxel_size, float sdf_trunc, bool space_carving = false);
    virtual ~VDBColoredVolume() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d> &points,
                   const std::vector<openvdb::Vec3i> &colors,
                   const Eigen::Vector3d &origin,
                   const std::function<float(float)> &weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d> &points,
                          const std::vector<openvdb::Vec3i> &colors,
                          const Eigen::Matrix4d &extrinsics,
                          const std::function<float(float)> &weighting_function) {
        const Eigen::Vector3d &origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, colors, origin, weighting_function);
    }

    /// @brief Extracts a TriangleMesh as the iso-surface in the actual volume
    [[nodiscard]] std::tuple<std::vector<Eigen::Vector3d>,
                             std::vector<Eigen::Vector3i>,
                             std::vector<Eigen::Vector3d>>
    ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;

public:
    /// OpenVDB Grids modeling the signed distance, weight and color
    openvdb::Vec3IGrid::Ptr colors_;

};

}  // namespace vdbfusion
