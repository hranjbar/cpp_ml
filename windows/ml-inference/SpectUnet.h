/*---------------------------------------------------------------------------*/
/*        Copyright (c) Siemens Medical Solutions USA, Inc.			         */
/*                            All Rights Reserved                            */
/*---------------------------------------------------------------------------*/
/*
	author:		Homayoon Ranjbar (homayoon.ranjbar@siemens-healthineers.com)
	date:		June 2023
*/

#pragma once

#include <iostream>
#include <filesystem>

#include "onnxruntime_cxx_api.h"

namespace ml
{
	namespace inference
	{
		class SpectUnet
		{
		public:
			SpectUnet(std::filesystem::path model_path);
			void summary();
			void infer(const std::string& inpVolFilename, const std::string& outVolFilename,
				const std::filesystem::path& inpVolDir, const std::filesystem::path& outVolDir);

		private:
			std::string inputName_, outputName_, instanceName_;
			std::vector<int64_t> inputDimensions_, outputDimensions_;

			Ort::Env env_;
			Ort::Session* session_;
			Ort::AllocatorWithDefaultOptions allocator_;
			Ort::RunOptions runOptions_;

			void fillInputTensorValues(std::vector<float>& i_vol, unsigned int slice_idx, unsigned int slice_stride,
				std::vector<float>& i_tsor_vals);
		};
	}
}