// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/geometry/lie.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <ceres/ceres.h>

namespace aliceVision {

namespace sfm {

class SL3Manifold : public ceres::Manifold
{
  public:
    ~SL3Manifold() override = default;

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        double* ptrBase = (double*)x;
        double* ptrResult = (double*)x_plus_delta;
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> H(ptrBase);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> HResult(ptrResult);

        Eigen::Matrix<double, 8, 1> algebra;
        for (int i = 0; i < 8; i++)
        {
            algebra[i] = delta[i];
        }
    

        HResult = H * SL3::expm(algebra);

        return true;
    }

    bool PlusJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 9, 8, Eigen::RowMajor>> J(jacobian);

        J.fill(0);

        J(0, 4) = 1;
        J(1, 3) = 1;
        J(2, 6) = 1;
        J(3, 2) = 1;
        J(4, 4) = -1;
        J(4, 5) = -1;
        J(5, 7) = 1;
        J(6, 0) = 1;
        J(7, 1) = 1;
        J(8, 5) = 1;

        return true;
    }

    bool Minus(const double* y, const double* x, double* delta) const override
    {
        throw std::invalid_argument("SL3::Manifold::Minus() should never be called");
    }

    bool MinusJacobian(const double* x, double* jacobian) const override
    {
        throw std::invalid_argument("SL3::Manifold::MinusJacobian() should never be called");
    }

    int AmbientSize() const override { return 9; }

    int TangentSize() const override { return 8; }
};

}  // namespace sfm

}  // namespace aliceVision
