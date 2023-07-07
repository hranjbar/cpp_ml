/*---------------------------------------------------------------------------*/
/*        Copyright (c) Siemens Medical Solutions USA, Inc.			         */
/*                            All Rights Reserved                            */
/*---------------------------------------------------------------------------*/
/*
	note:		Template class for multi-dimensional tensor container
	author:		Homayoon Ranjbar (homayoon.ranjbar@siemens-healthineers.com)
	date:		July 2023
*/

#pragma once

#include <iostream> 
#include <vector>
#include <cassert>

namespace ml
{
	template<typename T, int D>
	class Tensor
	{
	public:
		Tensor<T, D>() = default;

		Tensor<T, D>(std::vector<int> tensor_size)
		{
			assert(D == tensor_size.size());
			dimensions_ = tensor_size;
			data_ = std::vector<Tensor<T, D - 1>>(tensor_size.back());
			for (auto& sub : data_) {
				sub = Tensor<T, D - 1>(std::vector<int>(tensor_size.begin(), tensor_size.end() - 1));
			}
		}

	private:
		std::vector<int> dimensions_;
		std::vector<Tensor<T, D - 1>> data_;
	};

	template<typename T>
	class Tensor<T, 1>
	{
	public:
		Tensor<T, 1>() = default;

		Tensor<T, 1>(std::vector<int> scaler_tensor_size)
		{
			data_ = std::vector<T>(scaler_tensor_size.front());
		}
	private:
		std::vector<T> data_;
	};
}

