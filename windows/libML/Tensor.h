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
#include <filesystem>
#include <fstream>

namespace ml
{
	template<typename T, int D>
	class Tensor
	{
	public:
		Tensor<T, D>() = default;
		Tensor<T, D>(std::vector<int> tsor_dims)
		{
			assert(D == tsor_dims.size());
			data_ = std::vector<Tensor<T, D - 1>>(tsor_dims.back());	// last-order
			for (auto& sub : data_) {
				sub = Tensor<T, D - 1>(std::vector<int>(tsor_dims.begin(), tsor_dims.end() - 1));
			}
		}
		Tensor<T, D>(const std::initializer_list<Tensor<T, D - 1>>& il)
		{
			data_ = std::vector<Tensor<T, D - 1>>(il.size());
			int i = 0;
			for (const auto& el : il) {
				data_[i] = el;
				i++;
			}
		}

		Tensor<T, D - 1>& operator [] (int ix) { return data_[ix]; }

		int bytes() { return data_.size() * data_.front().bytes(); }
		void writeBin(std::ofstream& os) { for (auto& sub : data_) sub.writeBin(os); }
		void writeBin(std::filesystem::path& outputPath)
		{
			std::ofstream os(outputPath, std::ios::out | std::ios::binary);
			if (!os.is_open()) std::cout << "cannot open " << outputPath << std::endl;
			std::cout << "writing " << bytes() << " bytes\n";
			for (auto& sub : data_) sub.writeBin(os);
			os.close();
		}
		void readBin(std::ifstream& is) { for (auto& sub : data_) sub.readBin(is); }
		void readBin(std::filesystem::path inputPath)
		{
			std::ifstream is(inputPath, std::ios::in | std::ios::binary);
			if (!is.is_open()) std::cout << "cannot open " << inputPath << std::endl;
			std::cout << "reading " << bytes() << " bytes\n";
			for (auto& sub : data_) sub.readBin(is);
			is.close();
		}

		friend std::ostream& operator << (std::ostream& os, ml::Tensor<T, D>& tsor)
		{
			std::vector<Tensor<T, D - 1>>& v = tsor.data_;
			os << "[";
			for (int i = 0; i < v.size() - 1; ++i) os << v[i] << " ";
			os << v.back() << "]";
			return os;
		}

	private:
		std::vector<Tensor<T, D - 1>> data_;
	};

	template<typename T>
	class Tensor<T, 1>
	{
	public:
		Tensor<T, 1>() = default;
		Tensor<T, 1>(const int& array_size) { data_ = std::vector<T>(array_size); }
		Tensor<T, 1>(const std::vector<int>& tsor_dims) { data_ = std::vector<T>(tsor_dims.front()); }
		Tensor<T, 1>(std::initializer_list<T>&& il) { data_ = std::vector<T>(il); }

		T& operator [] (int ix) { return data_[ix]; }

		int bytes() { return data_.size() * sizeof T; }
		void writeBin(std::ofstream& os) { os.write(reinterpret_cast<char*>(data_.data()), bytes()); }
		void writeBin(std::filesystem::path& outputPath)
		{
			std::ofstream os(outputPath, std::ios::out | std::ios::binary);
			if (!os.is_open()) std::cout << "cannot open " << outputPath << std::endl;
			writeBin(os);
			os.close();
		}
		void readBin(std::ifstream& is) { is.read(reinterpret_cast<char*>(data_.data()), bytes()); }
		void readBin(std::filesystem::path& inputPath)
		{
			std::ifstream is(inputPath, std::ios::in | std::ios::binary);
			if (!is.is_open()) std::cout << "cannot open " << inputPath << std::endl;
			readBin(is);
			is.close();
		}

		friend std::ostream& operator << (std::ostream& os, ml::Tensor<T, 1>& tsor)
		{
			std::vector<T>& v = tsor.data_;
			os << "[";
			for (int i = 0; i < v.size() - 1; ++i) os << v[i] << " ";
			os << v.back() << "]";
			return os;
		}

	private:
		std::vector<T> data_;
	};
}

