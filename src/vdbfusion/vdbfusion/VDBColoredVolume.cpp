#include "VDBColoredVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace {

float ComputeSDF(const Eigen::Vector3d &origin,
                 const Eigen::Vector3d &point,
                 const Eigen::Vector3d &voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

Eigen::Vector3d GetVoxelCenter(const openvdb::Coord &voxel, const openvdb::math::Transform &xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return {v_wf.x(), v_wf.y(), v_wf.z()};
}

openvdb::Vec3i BlendColors(const openvdb::Vec3i &color1,
                           float weight1,
                           const openvdb::Vec3i &color2,
                           float weight2) {
    float weight_sum = weight1 + weight2;
    weight1 /= weight_sum;
    weight2 /= weight_sum;
    return {static_cast<int>(round(color1[0] * weight1 + color2[0] * weight2)),
            static_cast<int>(round(color1[1] * weight1 + color2[1] * weight2)),
            static_cast<int>(round(color1[2] * weight1 + color2[2] * weight2))};
}

}  // namespace

namespace vdbfusion {

VDBColoredVolume::VDBColoredVolume(float voxel_size, float sdf_trunc, bool space_carving /* = false*/)
    : VDBVolume(voxel_size,sdf_trunc,space_carving) {
    colors_ = openvdb::Vec3IGrid::create(openvdb::Vec3I(0.0f, 0.0f, 0.0f));
    colors_->setName("C(x): colors grid");
    colors_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    colors_->setGridClass(openvdb::GRID_UNKNOWN);
}


void VDBColoredVolume::Integrate(const std::vector<Eigen::Vector3d> &points,
                          const std::vector<openvdb::Vec3i> &colors,
                          const Eigen::Vector3d &origin,
                          const std::function<float(float)> &weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }
    bool has_colors = !colors.empty();
    if (has_colors && points.size() != colors.size()) {
        std::cerr << "PointCloud and ColorCloud provided do not have the same size\n";
        return;
    }

    VDBVolume::Integrate(points,origin,weighting_function);

    // Get some variables that are common to all rays
    const openvdb::math::Transform &xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Get the "unsafe" version of the grid accessors
    auto weights_acc = weights_->getUnsafeAccessor();
    auto colors_acc = colors_->getUnsafeAccessor();

    // Iterate points
    for (size_t i = 0; i < points.size(); ++i) {
        // Get the direction from the sensor origin to the point and normalize it
        const auto point = points[i];
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                if (has_colors) {
                    const auto color = colors_acc.getValue(voxel);
                    const auto new_color = BlendColors(color, last_weight, colors[i], weight);
                    colors_acc.setValue(voxel, new_color);
                }
            }
        } while (dda.step());
    }
}

}  // namespace vdbfusion
