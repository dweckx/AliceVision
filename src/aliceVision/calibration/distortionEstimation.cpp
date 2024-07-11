// This file is part of the AliceVision project.
// Copyright (c) 2021 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "distortionEstimation.hpp"

#include <aliceVision/system/Logger.hpp>
#include <aliceVision/sfm/bundle/manifolds/so3.hpp>

#include <ceres/ceres.h>

#include <cmath>

namespace aliceVision {
namespace calibration {

class CostLineDistanceRatio : public ceres::CostFunction
{
  public:
    CostLineDistanceRatio(double target)
      : _target(target)
    {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_dist_line11 = parameters[0];
        const double* parameter_dist_line12 = parameters[1];
        const double* parameter_dist_line21 = parameters[2];
        const double* parameter_dist_line22 = parameters[3];

        const double dist_line11 = parameter_dist_line11[0];
        const double dist_line12 = parameter_dist_line12[0];
        const double dist_line21 = parameter_dist_line21[0];
        const double dist_line22 = parameter_dist_line22[0];

        const double diff1 = dist_line12 - dist_line11;
        const double diff2 = dist_line22 - dist_line21;
        const double ratio = diff1 / diff2;

        double w = 10000.0;
        residuals[0] = w * (ratio - _target);

        if (jacobians == nullptr)
        {
            return true;
        }

        double d_ratio_d_diff1 = 1.0 / diff2;
        double d_ratio_d_diff2 = - diff1 / (diff2 * diff2);

        if (jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[0]);
            J(0, 0) = - w * (d_ratio_d_diff1);
        }

        if (jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[1]);
            J(0, 0) = w * (d_ratio_d_diff1);
        }

        if (jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[2]);
            J(0, 0) = - w * (d_ratio_d_diff2);
        }

        if (jacobians[3] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[3]);
            J(0, 0) = w * (d_ratio_d_diff2);
        }

        return true;
    }

  private:
    double _target;
};

class CostLineDistanceRatioSharedLine : public ceres::CostFunction
{
  public:
    CostLineDistanceRatioSharedLine(double target)
      : _target(target)
    {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_dist_line11 = parameters[0];
        const double* parameter_dist_line12 = parameters[1];
        const double* parameter_dist_line21 = parameters[1];
        const double* parameter_dist_line22 = parameters[2];

        const double dist_line11 = parameter_dist_line11[0];
        const double dist_line12 = parameter_dist_line12[0];
        const double dist_line21 = parameter_dist_line21[0];
        const double dist_line22 = parameter_dist_line22[0];

        const double diff1 = dist_line12 - dist_line11;
        const double diff2 = dist_line22 - dist_line21;
        const double ratio = diff1 / diff2;

        double w = 1.0;
        residuals[0] = w * (ratio - _target);

        if (jacobians == nullptr)
        {
            return true;
        }

        double d_ratio_d_diff1 = 1.0 / diff2;
        double d_ratio_d_diff2 = - diff1 / (diff2 * diff2);

        if (jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[0]);
            J(0, 0) = - w * (d_ratio_d_diff1);
        }

        if (jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[1]);
            J(0, 0) = w * (d_ratio_d_diff1) - w * (d_ratio_d_diff2);
        }

        if (jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[2]);
            J(0, 0) = w * (d_ratio_d_diff2);
        }

        return true;
    }

  private:
    double _target;
};

class CostLineAngle : public ceres::CostFunction
{
  public:
    CostLineAngle(double target)
      : _target(target)
    {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_angle_line = parameters[0];

        const double angle = parameter_angle_line[0];

        double dot = angle - _target;
        double w = 1.0;
        residuals[0] = w * dot;

        if (jacobians == nullptr)
        {
            return true;
        }

        if (jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[0]);
            J(0, 0) = w;
        }

        return true;
    }

  private:
    double _target;
};

class CostLine : public ceres::CostFunction
{
  public:
    CostLine(const std::shared_ptr<camera::Undistortion>& undistortion, const Vec2& pt, double offset)
      : _pt(pt),
        _undistortion(undistortion),
        _offset(offset)
    {
        set_num_residuals(1);

        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(undistortion->getParameters().size());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_angle_line = parameters[0];
        const double* parameter_dist_line = parameters[1];
        const double* parameter_offset = parameters[2];
        const double* parameter_disto = parameters[3];

        const double angle = parameter_angle_line[0];
        const double distanceToLine = parameter_dist_line[0];

        const double cangle = std::cos(angle + _offset);
        const double sangle = std::sin(angle + _offset);

        std::vector<double> cameraDistortionParams = _undistortion->getParameters();
        const std::size_t distortionSize = cameraDistortionParams.size();
        for (std::size_t idParam = 0; idParam < distortionSize; ++idParam)
        {
            cameraDistortionParams[idParam] = parameter_disto[idParam];
        }
        _undistortion->setParameters(cameraDistortionParams);

        Vec2 undistortionOffset;
        undistortionOffset.x() = parameter_offset[0];
        undistortionOffset.y() = parameter_offset[1];
        _undistortion->setOffset(undistortionOffset);

        const Vec2 ipt = _undistortion->undistort(_pt);
        const double pa = _undistortion->getPixelAspectRatio();
        const double ny = ipt.y() / pa;
        const double w = 1.0;
        
        residuals[0] = w * (cangle * ipt.x() + sangle * ny - distanceToLine);

        if (jacobians == nullptr)
        {
            return true;
        }

        if (jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[0]);
            J(0, 0) = w * (ipt.x() * -sangle + ny * cangle);
        }

        if (jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[1]);
            J(0, 0) = -w;
        }

        if (jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> J(jacobians[2]);

            Eigen::Matrix<double, 1, 2> Jline;
            Jline(0, 0) = cangle;
            Jline(0, 1) = sangle / pa;

            J = w * Jline * _undistortion->getDerivativeUndistortWrtOffset(_pt);
        }

        if (jacobians[3] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[3], 1, distortionSize);

            Eigen::Matrix<double, 1, 2> Jline;
            Jline(0, 0) = cangle;
            Jline(0, 1) = sangle / pa;

            J = w * Jline * _undistortion->getDerivativeUndistortWrtParameters(_pt);
        }

        return true;
    }

  private:
    std::shared_ptr<camera::Undistortion> _undistortion;
    Vec2 _pt;
    double _offset;
};

class CostPoint : public ceres::CostFunction
{
public:
    CostPoint(std::shared_ptr<camera::Undistortion> & undistortion, const Vec2& ptUndistorted, const Vec2 &ptDistorted)
        : _ptUndistorted(ptUndistorted)
        , _ptDistorted(ptDistorted)
        , _undistortion(undistortion)
    {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(undistortion->getUndistortionParametersCount());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_center = parameters[0];
        const double* parameter_disto = parameters[1];

        const int undistortionSize = _undistortion->getUndistortionParametersCount();

        //Read parameters and update camera
        std::vector<double> undistortionParams(undistortionSize);

        for (int idParam = 0; idParam < undistortionSize; idParam++)
        {
            undistortionParams[idParam] = parameter_disto[idParam];
        }
        _undistortion->setParameters(undistortionParams);

        Vec2 offset;
        offset.x() = parameter_center[0];
        offset.y() = parameter_center[1];
        _undistortion->setOffset(offset);

        //Estimate measure
        const Vec2 ipt = _undistortion->undistort(_ptDistorted);

        const Vec2 hsize = (_undistortion->getSize() * 0.5);
        Vec2 npt = (_ptDistorted - hsize);
        npt.x() = npt.x() / hsize.x();
        npt.y() = npt.y() / hsize.y();

        const double w1 = std::max(std::abs(npt.x()), std::abs(npt.y()));
        const double w = w1 * w1;

        residuals[0] = w * (ipt.x() - _ptUndistorted.x());
        residuals[1] = w * (ipt.y() - _ptUndistorted.y());

        if(jacobians == nullptr)
        {
            return true;
        }


        if(jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J(jacobians[0]);

            J = w * _undistortion->getDerivativeUndistortWrtOffset(_ptDistorted);
        }

        if(jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[1], 2, undistortionSize);

            J = w * _undistortion->getDerivativeUndistortWrtParameters(_ptDistorted);
        }

        return true;
    }

private:
    std::shared_ptr<camera::Undistortion> _undistortion;
    Vec2 _ptUndistorted;
    Vec2 _ptDistorted;
};

class CostPointGeometry : public ceres::CostFunction
{
public:
    CostPointGeometry(std::shared_ptr<camera::Undistortion> & undistortion, const Vec2& ptUndistorted, const Vec2 &ptDistorted)
        : _ptUndistorted(ptUndistorted)
        , _ptDistorted(ptDistorted)
        , _undistortion(undistortion)
    {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(undistortion->getUndistortionParametersCount());
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(9);
        mutable_parameter_block_sizes()->push_back(3);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_center = parameters[0];
        const double* parameter_disto = parameters[1];
        const double* parameter_scale = parameters[2];
        const double* parameter_offsetx = parameters[3];
        const double* parameter_offsety = parameters[4];
        const double* parameter_rotation = parameters[5];
        const double* parameter_translation = parameters[6];

        const double scale = *parameter_scale;
        const double offsetx = *parameter_offsetx;
        const double offsety = *parameter_offsety;

        const Eigen::Map<const SO3::Matrix> R(parameter_rotation);
        const Eigen::Map<const Eigen::Vector3d> t(parameter_translation);

        const int undistortionSize = _undistortion->getUndistortionParametersCount();


        //Read parameters and update camera
        std::vector<double> undistortionParams(undistortionSize);

        for (int idParam = 0; idParam < undistortionSize; idParam++)
        {
            undistortionParams[idParam] = parameter_disto[idParam];
        }
        _undistortion->setParameters(undistortionParams);

        Vec2 offset;
        offset.x() = parameter_center[0];
        offset.y() = parameter_center[1];
        _undistortion->setOffset(offset);

        const double pa = _undistortion->getPixelAspectRatio();

        const Vec2 upt = _undistortion->undistort(_ptDistorted);

        //Estimate measure
        const Vec2 ipt = _ptUndistorted;
                
        Vec3 ipt3;
        ipt3.x() = ipt.x();
        ipt3.y() = ipt.y();
        ipt3.z() = 0.0;

        const Vec3 tpt = R * ipt3 + t;
        
        Vec2 projected = tpt.head(2) / tpt(2);
        Vec2 scaled;

        scaled.x() = scale * projected.x() + offsetx;
        scaled.y() = pa * scale * projected.y() + offsety;
                
        const double w = 1 + ipt.norm();

        residuals[0] = w * (upt.x() - scaled.x());
        residuals[1] = w * (upt.y() - scaled.y());

        if(jacobians == nullptr)
        {
            return true;
        }

        

        if(jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J(jacobians[0]);

            J = w * _undistortion->getDerivativeUndistortWrtOffset(_ptDistorted);
        }

        if(jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[1], 2, undistortionSize);

            J = w * _undistortion->getDerivativeUndistortWrtParameters(_ptDistorted);
        }

        if (jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[2], 2, 1);
            
            Vec2 d_scaled_d_scale;
            d_scaled_d_scale(0) = projected.x();
            d_scaled_d_scale(1) = pa * projected.y();
            
            J = -w * d_scaled_d_scale;
        }

        if (jacobians[3] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[3], 2, 1);
            
            Vec2 d_scaled_d_offsetx;
            d_scaled_d_offsetx(0) = 1.0;
            d_scaled_d_offsetx(1) = 0.0;
            
            J = -w * d_scaled_d_offsetx;
        }

        if (jacobians[4] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[4], 2, 1);
            
            Vec2 d_scaled_d_offsety;
            d_scaled_d_offsety(0) = 0.0;
            d_scaled_d_offsety(1) = 1.0;
            
            J = -w * d_scaled_d_offsety;
        }

        if (jacobians[5] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[5]);
            
            Eigen::Matrix<double, 2, 3> d_projected_d_tpt;
            d_projected_d_tpt(0, 0) = 1.0 / tpt.z();
            d_projected_d_tpt(0, 1) = 0.0;
            d_projected_d_tpt(0, 2) = - tpt.x() / (tpt.z() * tpt.z());
            d_projected_d_tpt(1, 0) = 0.0;
            d_projected_d_tpt(1, 1) = 1.0 / tpt.z();
            d_projected_d_tpt(1, 2) = - tpt.y() / (tpt.z() * tpt.z());

            Eigen::Matrix<double, 3, 2> d_ipt3_d_ipt;
            d_ipt3_d_ipt(0, 0) = 1;
            d_ipt3_d_ipt(0, 1) = 0;
            d_ipt3_d_ipt(1, 0) = 0;
            d_ipt3_d_ipt(1, 1) = 1;
            d_ipt3_d_ipt(2, 0) = 0;
            d_ipt3_d_ipt(2, 1) = 0;

            Eigen::Matrix<double, 2, 2> d_scaled_d_projected;
            d_scaled_d_projected(0, 0) = scale;
            d_scaled_d_projected(0, 1) = 0;
            d_scaled_d_projected(1, 0) = 0;
            d_scaled_d_projected(1, 1) = pa * scale;

            J = - w * d_scaled_d_projected * d_projected_d_tpt * getJacobian_AB_wrt_A<3, 3, 1>(R, ipt3) * getJacobian_AB_wrt_A<3, 3, 3>(Eigen::Matrix3d::Identity(), R);
        }

        if (jacobians[6] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[6]);
            
            Eigen::Matrix<double, 2, 3> d_projected_d_tpt;
            d_projected_d_tpt(0, 0) = 1.0 / tpt.z();
            d_projected_d_tpt(0, 1) = 0.0;
            d_projected_d_tpt(0, 2) = - tpt.x() / (tpt.z() * tpt.z());
            d_projected_d_tpt(1, 0) = 0.0;
            d_projected_d_tpt(1, 1) = 1.0 / tpt.z();
            d_projected_d_tpt(1, 2) = - tpt.y() / (tpt.z() * tpt.z());

            Eigen::Matrix<double, 3, 2> d_ipt3_d_ipt;
            d_ipt3_d_ipt(0, 0) = 1;
            d_ipt3_d_ipt(0, 1) = 0;
            d_ipt3_d_ipt(1, 0) = 0;
            d_ipt3_d_ipt(1, 1) = 1;
            d_ipt3_d_ipt(2, 0) = 0;
            d_ipt3_d_ipt(2, 1) = 0;

            Eigen::Matrix<double, 2, 2> d_scaled_d_projected;
            d_scaled_d_projected(0, 0) = scale;
            d_scaled_d_projected(0, 1) = 0;
            d_scaled_d_projected(1, 0) = 0;
            d_scaled_d_projected(1, 1) = pa * scale;

            J = - w * d_scaled_d_projected * d_projected_d_tpt;
        }

        return true;
    }

private:
    std::shared_ptr<camera::Undistortion> _undistortion;
    Vec2 _ptUndistorted;
    Vec2 _ptDistorted;
};

bool estimate(std::shared_ptr<camera::Undistortion> undistortionToEstimate,
              Statistics& statistics,
              std::vector<LineWithPoints>& lines,
              const std::vector<calibration::SizeConstraint>& constraints,
              const bool lockCenter,
              const bool lockAngles,
              const std::vector<bool>& lockDistortions)
{
    if (!undistortionToEstimate)
    {
        return false;
    }

    if (lines.empty())
    {
        return false;
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction = nullptr;

    std::vector<double> undistortionParameters = undistortionToEstimate->getParameters();
    Vec2 undistortionOffset = undistortionToEstimate->getOffset();
    const std::size_t countUndistortionParams = undistortionParameters.size();

    if (lockDistortions.size() != countUndistortionParams)
    {
        ALICEVISION_LOG_ERROR("Invalid number of distortion parameters (lockDistortions=" << lockDistortions.size() << ", countDistortionParams="
                                                                                          << countUndistortionParams << ").");
        return false;
    }

    double* ptrUndistortionParameters = &undistortionParameters[0];
    double* center = &undistortionOffset.x();

    // Add off center parameter
    problem.AddParameterBlock(center, 2);
    if (lockCenter)
    {
        problem.SetParameterBlockConstant(center);
    }

    // Add distortion parameter
    problem.AddParameterBlock(ptrUndistortionParameters, countUndistortionParams);

    // Check if all distortions are locked
    bool allLocked = true;
    for (bool lock : lockDistortions)
    {
        if (!lock)
        {
            allLocked = false;
        }
    }

    if (allLocked)
    {
        problem.SetParameterBlockConstant(ptrUndistortionParameters);
    }
    else
    {
        // At least one parameter is not locked

        std::vector<int> constantDistortions;
        for (int idParamDistortion = 0; idParamDistortion < lockDistortions.size(); ++idParamDistortion)
        {
            if (lockDistortions[idParamDistortion])
            {
                constantDistortions.push_back(idParamDistortion);
            }
        }

        if (!constantDistortions.empty())
        {
            ceres::SubsetManifold* subsetManifold = new ceres::SubsetManifold(countUndistortionParams, constantDistortions);
            problem.SetManifold(ptrUndistortionParameters, subsetManifold);
        }
    }

    std::map<int, double> anglePerGroup;

    for (auto& l : lines)
    {
        if (anglePerGroup.find(l.groupId) == anglePerGroup.end())
        {
            anglePerGroup[l.groupId] = l.angle;
            problem.AddParameterBlock(&anglePerGroup[l.groupId], 1);
        }

        problem.AddParameterBlock(&l.dist, 1);

        for (Vec2 pt : l.points)
        {
            ceres::CostFunction* costFunction = new CostLine(undistortionToEstimate, pt, l.angleOffset);
            problem.AddResidualBlock(costFunction, lossFunction, &anglePerGroup[l.groupId], &l.dist, center, ptrUndistortionParameters);
        }
    }

    if (lockAngles)
    {
        for (auto & [groupId, angle] : anglePerGroup)
        {
            problem.SetParameterBlockConstant(&angle);
        }
    }

    for (auto & c : constraints)
    {
        LineWithPoints & l11 = lines[c.firstPair.first];
        LineWithPoints & l12 = lines[c.firstPair.second];
        LineWithPoints & l21 = lines[c.secondPair.first];
        LineWithPoints & l22 = lines[c.secondPair.second];

        double s1 = l12.step - l11.step;
        double s2 = l22.step - l21.step;
        double ratio = s1 / s2;

        if (c.secondPair.first == c.firstPair.second)
        {
            ceres::CostFunction* costFunction = new CostLineDistanceRatioSharedLine(ratio);
            problem.AddResidualBlock(costFunction, lossFunction, &l11.dist, &l12.dist, &l22.dist);
        }
        else 
        {
            ceres::CostFunction* costFunction = new CostLineDistanceRatio(ratio);
            problem.AddResidualBlock(costFunction, lossFunction, &l11.dist, &l12.dist, &l21.dist, &l22.dist);
        }
    }


    ceres::Solver::Options options;
    options.use_inner_iterations = true;
    options.max_num_iterations = 100;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ALICEVISION_LOG_TRACE(summary.FullReport());

    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_ERROR("Lens calibration estimation failed.");
        return false;
    }

    undistortionToEstimate->setOffset(undistortionOffset);
    undistortionToEstimate->setParameters(undistortionParameters);
    std::vector<double> errors;

    for (auto& l : lines)
    {
        l.angle = anglePerGroup[l.groupId];
    }

   
    for (auto & c : constraints)
    {
        LineWithPoints & l11 = lines[c.firstPair.first];
        LineWithPoints & l12 = lines[c.firstPair.second];
        LineWithPoints & l21 = lines[c.secondPair.first];
        LineWithPoints & l22 = lines[c.secondPair.second];

        double s1 = l12.step - l11.step;
        double s2 = l22.step - l21.step;
        double d1 = l12.dist - l11.dist;
        double d2 = l22.dist - l21.dist;
        
        if (std::abs(s1/s2 - d1/d2) > 1e-1) 
        {
            ALICEVISION_LOG_INFO("Some constraint is not respected");
            std::cout << s1 << " " << s2 << " " << d1 << " " << d2 << std::endl;
            std::cout << s1 / s2 << " " << d1 / d2 << std::endl;
            std::cout << std::abs(s1/s2 - d1/d2) << std::endl;
        }
    }

    

    for (auto& l : lines)
    {
        const double sangle = std::sin(l.angle + l.angleOffset);
        const double cangle = std::cos(l.angle + l.angleOffset);

        for (const Vec2& pt : l.points)
        {
            const Vec2 ipt = undistortionToEstimate->undistort(pt);
            const double pa = undistortionToEstimate->getPixelAspectRatio();
            const double ny = ipt.y() / pa;
            const double res = (cangle * ipt.x() + sangle * ny - l.dist);
            errors.push_back(std::abs(res));
        }
    }
 
    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    const double sqSum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    const double stddev = std::sqrt(sqSum / errors.size() - mean * mean);
    std::sort(errors.begin(), errors.end());
    const double median = errors[errors.size() / 2];
    const double max = errors[errors.size() - 1];
    const double lastDecile = errors[errors.size() * 0.9];

    statistics.mean = mean;
    statistics.stddev = stddev;
    statistics.median = median;
    statistics.max = max;
    statistics.lastDecile = lastDecile;

    return true;
}

bool estimate(std::shared_ptr<camera::Undistortion> undistortionToEstimate,
              Statistics& statistics,
              const std::vector<PointPair>& pointpairs,
              const bool lockCenter,
              const std::vector<bool>& lockDistortions)
{
    if (!undistortionToEstimate)
    {
        return false;
    }

    if (pointpairs.empty())
    {
        return false;
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction = new ceres::HuberLoss(2.0);

    std::vector<double> undistortionParameters = undistortionToEstimate->getParameters();
    Vec2 undistortionOffset = undistortionToEstimate->getOffset();

    const std::size_t countUndistortionParams = undistortionParameters.size();

    if (lockDistortions.size() != countUndistortionParams)
    {
        ALICEVISION_LOG_ERROR("Invalid number of distortion parameters (lockDistortions=" << lockDistortions.size() << ", countDistortionParams="
                                                                                          << countUndistortionParams << ").");
        return false;
    }

    double* ptrUndistortionParameters = &undistortionParameters[0];
    double* center = &undistortionOffset.x();

    // Add off center parameter
    problem.AddParameterBlock(center, 2);
    if (lockCenter)
    {
        problem.SetParameterBlockConstant(center);
    }

    // Add distortion parameter
    problem.AddParameterBlock(ptrUndistortionParameters, countUndistortionParams);

    // Check if all distortions are locked
    bool allLocked = true;
    for (bool lock : lockDistortions)
    {
        if (!lock)
        {
            allLocked = false;
        }
    }

    if (allLocked)
    {
        problem.SetParameterBlockConstant(ptrUndistortionParameters);
    }
    else
    {
        // At least one parameter is not locked

        std::vector<int> constantDistortions;
        for (int idParamDistortion = 0; idParamDistortion < lockDistortions.size(); ++idParamDistortion)
        {
            if (lockDistortions[idParamDistortion])
            {
                constantDistortions.push_back(idParamDistortion);
            }
        }

        if (!constantDistortions.empty())
        {
            ceres::SubsetManifold* subsetManifold = new ceres::SubsetManifold(countUndistortionParams, constantDistortions);
            problem.SetManifold(ptrUndistortionParameters, subsetManifold);
        }
    }

    for (auto& ppt : pointpairs)
    {
        ceres::CostFunction* costFunction = new CostPoint(undistortionToEstimate, ppt.undistortedPoint, ppt.distortedPoint);
        problem.AddResidualBlock(costFunction, lossFunction, center, ptrUndistortionParameters);
    }

    ceres::Solver::Options options;
    options.use_inner_iterations = true;
    options.max_num_iterations = 100;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ALICEVISION_LOG_TRACE(summary.FullReport());

    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_ERROR("Lens calibration estimation failed.");
        return false;
    }

    undistortionToEstimate->setOffset(undistortionOffset);
    undistortionToEstimate->setParameters(undistortionParameters);
    std::vector<double> errors;

    for (auto& ppt : pointpairs)
    {
        const Vec2 ipt = undistortionToEstimate->undistort(ppt.distortedPoint);
        const double res = (ipt - ppt.undistortedPoint).norm();
        errors.push_back(res);
    }

    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    const double sqSum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    const double stddev = std::sqrt(sqSum / errors.size() - mean * mean);
    std::sort(errors.begin(), errors.end());
    const double median = errors[errors.size() / 2];
    const double max = errors[errors.size() - 1];
    const double lastDecile = errors[errors.size() * 0.9];

    statistics.mean = mean;
    statistics.stddev = stddev;
    statistics.median = median;
    statistics.max = max;
    statistics.lastDecile = lastDecile;
    
    return true;
}

bool estimate(std::shared_ptr<camera::Undistortion> undistortionToEstimate,
              Statistics& statistics,
              const std::vector<PointPair>& pointpairs,
              const bool lockCenter,
              const std::vector<bool>& lockDistortions,
              Eigen::Matrix3d & R, 
              Eigen::Vector3d & t,
              double & scale)
{
    if (!undistortionToEstimate)
    {
        return false;
    }

    if (pointpairs.empty())
    {
        return false;
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction = nullptr;

    std::vector<double> undistortionParameters = undistortionToEstimate->getParameters();
    Vec2 undistortionOffset = undistortionToEstimate->getOffset();

    const std::size_t countUndistortionParams = undistortionParameters.size();

    if (lockDistortions.size() != countUndistortionParams)
    {
        ALICEVISION_LOG_ERROR("Invalid number of distortion parameters (lockDistortions=" << lockDistortions.size() << ", countDistortionParams="
                                                                                          << countUndistortionParams << ").");
        return false;
    }

    double* ptrUndistortionParameters = &undistortionParameters[0];
    double* center = &undistortionOffset.x();

    // Add off center parameter
    problem.AddParameterBlock(center, 2);
    if (lockCenter)
    {
        problem.SetParameterBlockConstant(center);
    }

    // Add distortion parameter
    problem.AddParameterBlock(ptrUndistortionParameters, countUndistortionParams);

    // Check if all distortions are locked
    bool allLocked = true;
    for (bool lock : lockDistortions)
    {
        if (!lock)
        {
            allLocked = false;
        }
    }

    if (allLocked)
    {
        problem.SetParameterBlockConstant(ptrUndistortionParameters);
    }
    else
    {
        // At least one parameter is not locked

        std::vector<int> constantDistortions;
        for (int idParamDistortion = 0; idParamDistortion < lockDistortions.size(); ++idParamDistortion)
        {
            if (lockDistortions[idParamDistortion])
            {
                constantDistortions.push_back(idParamDistortion);
            }
        }

        if (!constantDistortions.empty())
        {
            ceres::SubsetManifold* subsetManifold = new ceres::SubsetManifold(countUndistortionParams, constantDistortions);
            problem.SetManifold(ptrUndistortionParameters, subsetManifold);
        }
    }

    double offsetx = undistortionToEstimate->getCenter().x();
    double offsety = undistortionToEstimate->getCenter().y();
    
    problem.AddParameterBlock(&scale, 1);
    problem.AddParameterBlock(&offsetx, 1);
    problem.AddParameterBlock(&offsety, 1);
    
    problem.SetParameterLowerBound(&offsetx, 0, offsetx - 100.0);
    problem.SetParameterUpperBound(&offsetx, 0, offsetx + 100.0);
    problem.SetParameterLowerBound(&offsety, 0, offsety - 100.0);
    problem.SetParameterUpperBound(&offsety, 0, offsety + 100.0);

    SO3::Matrix Rest = R;
    Eigen::Vector3d Test = t;

    problem.AddParameterBlock(Rest.data(), 9);
    problem.AddParameterBlock(Test.data(), 3);

    problem.SetManifold(Rest.data(), new sfm::SO3Manifold);

    for (auto& ppt : pointpairs)
    {
        ceres::CostFunction* costFunction = new CostPointGeometry(undistortionToEstimate, ppt.undistortedPoint, ppt.distortedPoint);
        problem.AddResidualBlock(costFunction, lossFunction, center, ptrUndistortionParameters, &scale, &offsetx, &offsety, Rest.data(), Test.data());
    }

    ceres::Solver::Options options;
    options.use_inner_iterations = true;
    options.max_num_iterations = 1000;
    /*options.function_tolerance = 1e-24;
    options.parameter_tolerance = 1e-24;
    options.gradient_tolerance = 1e-24;*/
    options.logging_type = ceres::SILENT;   
    
        
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    ALICEVISION_LOG_TRACE(summary.FullReport());

    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_ERROR("Lens calibration estimation failed.");
        return false;
    }

    undistortionToEstimate->setOffset(undistortionOffset);
    undistortionToEstimate->setParameters(undistortionParameters);
    std::vector<double> errors;
    const double pa = undistortionToEstimate->getPixelAspectRatio();

    for (auto& ppt : pointpairs)
    {
        const Vec2 ipt = ppt.undistortedPoint;
        
        Vec3 ipt3;
        ipt3.x() = ipt.x();
        ipt3.y() = ipt.y();
        ipt3.z() = 0.0;

        const Vec3 tpt = Rest * ipt3 + Test;
        
        Vec2 projected = tpt.head(2) / tpt(2);
        Vec2 scaled;
        scaled.x() = scale * projected.x() + offsetx;
        scaled.y() = pa * scale * projected.y() + offsety;
    
        const double res = (undistortionToEstimate->undistort(ppt.distortedPoint) - scaled).norm();
        
        errors.push_back(res);
    }

    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    const double sqSum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    const double stddev = std::sqrt(sqSum / errors.size() - mean * mean);
    std::sort(errors.begin(), errors.end());
    const double median = errors[errors.size() / 2];
    const double max = errors[errors.size() - 1];
    const double lastDecile = errors[errors.size() * 0.9];

    statistics.mean = mean;
    statistics.stddev = stddev;
    statistics.median = median;
    statistics.max = max;
    statistics.lastDecile = lastDecile;

    R = Rest;
    t = Test;

    std::cout << scale << std::endl;
    std::cout << offsetx << std::endl;
    std::cout << offsety << std::endl;
    std::cout << R << std::endl;
    std::cout << t << std::endl;
    
    return true;
}


}  // namespace calibration
}  // namespace aliceVision
