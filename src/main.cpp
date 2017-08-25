#include <gflags/gflags.h>

#include <NN_Agent.h>
#include <random>
#include <chrono>
#include <thread>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

//Common flags
DEFINE_int32(train, 1, "Whether we want to train the net or generate new data");
DEFINE_string(weights_gen, "", "Trained weights to load into the generator net (.caffemodel)");
DEFINE_int32(z_dim, 64, "Dimension of the input for generator");
DEFINE_int32(h_dim, 64, "Dimension of the encoded features for discriminator");
DEFINE_int32(hidden_dim, 64, "Internal dimension in nets (n in the paper)");
DEFINE_int32(image_size, 64, "Size of the images (16, 32, 64, 128)");
DEFINE_int32(batch_size, 16, "Number of samples in one pass");

//Training flags
DEFINE_string(solver_gen, "solver_generator.prototxt", "Caffe solver file for generator");
DEFINE_string(solver_discr, "solver_discriminator.prototxt", "Caffe solver file for discriminator");
DEFINE_string(weights_discr, "", "Trained weights to load into the discriminator net (.caffemodel)");
DEFINE_string(snapshot_generator, "", "Snaphsot file to resume generator training (*.solverstate)");
DEFINE_string(snapshot_discriminator, "", "Snaphsot file to resume discriminator training (*.solverstate)");
DEFINE_string(snapshot_k_t, "", "Snapshot file to set starting k_t value");
DEFINE_string(preview_generator, "preview_values.csv", "File in which saving the generator input to see training evolution");
DEFINE_int32(number_batch_loaded, 50, "Number of batch of images loaded in the memory at the same time.");
DEFINE_int32(start_epoch, 0, "Epoch number to start training");
DEFINE_int32(end_epoch, 50, "Epoch number to train on");
DEFINE_string(training_dataset, "../Data/CelebA", "Dataset in which the training images are");

//Testing flags
DEFINE_int32(save_img, 0, "1 to save the generated faces as bmp, 0 to just display them");
DEFINE_int32(num_gen, 10, "Number of generated faces. -1 to generate indefinitely until program is terminated");

/**
* \brief Load some images in a destination vector. If the image is too big, only the 128x128 center pixels are kept
* \param files_path Path of the files
* \param files_indices All the indices in the files_path vector of the files we want to load
* \param destination Destination vector. Cleaned at the begining
* \param image_size Desired size of the images
*/
void LoadImagesFromFiles(const std::vector<std::string> &files_path, const std::vector<int> files_indices, std::vector<std::vector<float> > *destination, const int &image_size)
{
	destination->clear();

	for (int i = 0; i < files_indices.size(); ++i)
	{
		cv::Mat img, img_crop, img_float;
		img = cv::imread(files_path[files_indices[i]]);

		cv::Rect img_ROI = cv::Rect((img.cols - 128) / 2, (img.rows - 128) / 2, 128, 128) & cv::Rect(0, 0, img.cols, img.rows);

		img(img_ROI).copyTo(img_crop);

		if (img_crop.cols != image_size || img_crop.rows != image_size)
		{
			cv::resize(img_crop, img_crop, cv::Size(image_size, image_size), 0.0, 0.0, cv::InterpolationFlags::INTER_LINEAR);
		}

		img_crop.convertTo(img_float, CV_32FC3);
		img_float /= 255.0f;

		std::vector<cv::Mat> channels;
		cv::split(img_float, channels);

		std::vector<float> current_image;
		current_image.reserve(FLAGS_image_size * FLAGS_image_size * 3);

		current_image.insert(current_image.end(), (float*)channels[0].datastart, (float*)channels[0].dataend);
		current_image.insert(current_image.end(), (float*)channels[1].datastart, (float*)channels[1].dataend);
		current_image.insert(current_image.end(), (float*)channels[2].datastart, (float*)channels[2].dataend);
		destination->push_back(current_image);
	}
}

/**
* \brief Create generator prototxt and save it in current directory as "generator.prototxt"
* \param batch_size Size of one batch in images
* \param z_dim Dimension of generator input (z in paper)
* \param hidden_dim Number of channels in hidden layers (n in paper)
* \param image_size Desired size for output images (16, 32, 64, 128 ...)
*/
void CreateGeneratorPrototxt(const int &batch_size, const int &z_dim, const int &hidden_dim, const int &image_size)
{
	std::ofstream prototxt("generator.prototxt");

	//Net header
	prototxt
		<< "name: \"Generator\"" << "\n"
		<< "force_backward: true" << "\n"
		<< "\n";

	//Input layer
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Input\"" << "\n"
		<< "\t" << "type: \"Input\"" << "\n"
		<< "\t" << "top: \"generator_z\"" << "\n"
		<< "\t" << "input_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << z_dim << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Layers to transform the input vector into small images
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"generator_ip1\"" << "\n"
		<< "\t" << "type: \"InnerProduct\"" << "\n"
		<< "\t" << "bottom: \"generator_z\"" << "\n"
		<< "\t" << "top: \"generator_ip1\"" << "\n"
		<< "\t" << "inner_product_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << hidden_dim * 8 * 8 << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"generator_reshape_ip1\"" << "\n"
		<< "\t" << "type: \"Reshape\"" << "\n"
		<< "\t" << "bottom: \"generator_ip1\"" << "\n"
		<< "\t" << "top: \"generator_size_8\"" << "\n"
		<< "\t" << "reshape_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << hidden_dim << "\n"
		<< "\t" << "\t" << "\t" << "dim: 8" << "\n"
		<< "\t" << "\t" << "\t" << "dim: 8" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Add some convolution/upsampling until the dimension is the desired one
	int current_size = 8;

	while (current_size <= image_size)
	{
		//First convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"generator_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"generator_size_" << current_size << "\"" << "\n"
			<< "\t" << "top: \"generator_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"generator_elu_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"generator_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"generator_conv_" << current_size << "_1\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Second convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"generator_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"generator_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"generator_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"generator_elu_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"generator_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "top: \"generator_conv_" << current_size << "_2\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Upsampling
		if (current_size != image_size)
		{
			prototxt
				<< "layer {" << "\n"
				<< "\t" << "name: \"generator_upsampling_" << current_size << "\"" << "\n"
				<< "\t" << "type: \"Deconvolution\"" << "\n"
				<< "\t" << "bottom: \"generator_conv_" << current_size << "_2\"" << "\n"
				<< "\t" << "top: \"generator_size_" << 2 * current_size << "\"" << "\n"
				<< "\t" << "convolution_param {" << "\n"
				<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
				<< "\t" << "\t" << "group: " << hidden_dim << "\n"
				<< "\t" << "\t" << "kernel_size: 2" << "\n"
				<< "\t" << "\t" << "stride: 2" << "\n"
				<< "\t" << "\t" << "pad: 0" << "\n"
				<< "\t" << "\t" << "weight_filler {" << "\n"
				<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
				<< "\t" << "\t" << "\t" << "value: 1" << "\n"
				<< "\t" << "\t" << "}" << "\n"
				<< "\t" << "\t" << "bias_term: false" << "\n"
				<< "\t" << "}" << "\n"
				<< "\t" << "param {" << "\n"
				<< "\t" << "\t" << "lr_mult: 0" << "\n"
				<< "\t" << "\t" << "decay_mult: 0" << "\n"
				<< "\t" << "}" << "\n"
				<< "}" << "\n"
				<< "\n";
		}
		current_size *= 2;
	}

	current_size /= 2;

	//Last layer to get a 3 channels image
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"generator_last_conv\"" << "\n"
		<< "\t" << "type: \"Convolution\"" << "\n"
		<< "\t" << "bottom: \"generator_conv_" << current_size << "_2\"" << "\n"
		<< "\t" << "top: \"generated_image\"" << "\n"
		<< "\t" << "convolution_param {" << "\n"
		<< "\t" << "\t" << "num_output: 3" << "\n"
		<< "\t" << "\t" << "kernel_size: 3" << "\n"
		<< "\t" << "\t" << "stride: 1" << "\n"
		<< "\t" << "\t" << "pad: 1" << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt.close();
}

/**
* \brief Create discriminator prototxt and save it in current directory as "discriminator.prototxt"
* \param batch_size Size of one batch in images
* \param h_dim Dimension of encoded images (h in paper)
* \param hidden_dim Number of channels in hidden layers (n in paper)
* \param image_size Size of input/output images (16, 32, 64, 128 ...)
*/
void CreateDiscriminatorPrototxt(const int &batch_size, const int &h_dim, const int &hidden_dim, const int &image_size)
{
	std::ofstream prototxt("discriminator.prototxt");

	//Net header
	prototxt
		<< "name: \"Encoder/Decoder\"" << "\n"
		<< "force_backward: true" << "\n"
		<< "\n";

	//Input layer
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Input\"" << "\n"
		<< "\t" << "type: \"Input\"" << "\n"
		<< "\t" << "top: \"image_input\"" << "\n"
		<< "\t" << "input_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: 3" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Layer to transform the 3 channels input image into an hidden_dim channels one
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"encoder_conv_init\"" << "\n"
		<< "\t" << "type: \"Convolution\"" << "\n"
		<< "\t" << "bottom: \"image_input\"" << "\n"
		<< "\t" << "top: \"encoder_size_" << image_size << "\"" << "\n"
		<< "\t" << "convolution_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
		<< "\t" << "\t" << "kernel_size: 3" << "\n"
		<< "\t" << "\t" << "stride: 1" << "\n"
		<< "\t" << "\t" << "pad: 1" << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Add some convolution/subsampling until the dimension is 8x8
	int current_size = image_size;
	int current_hidden_dim = hidden_dim;

	while (current_size >= 8)
	{
		//First convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"encoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"encoder_size_" << current_size << "\"" << "\n"
			<< "\t" << "top: \"encoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << current_hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"encoder_elu_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"encoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"encoder_conv_" << current_size << "_1\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Second convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"encoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"encoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"encoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << current_hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"encoder_elu_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"encoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "top: \"encoder_conv_" << current_size << "_2\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Subsampling
		if (current_size != 8)
		{
			prototxt
				<< "layer {" << "\n"
				<< "\t" << "name: \"encoder_subsampling_" << current_size << "\"" << "\n"
				<< "\t" << "type: \"Convolution\"" << "\n"
				<< "\t" << "bottom: \"encoder_conv_" << current_size << "_2\"" << "\n"
				<< "\t" << "top: \"encoder_size_" << current_size / 2 << "\"" << "\n"
				<< "\t" << "convolution_param {" << "\n"
				<< "\t" << "\t" << "num_output: " << current_hidden_dim << "\n"
				<< "\t" << "\t" << "group: " << current_hidden_dim << "\n"
				<< "\t" << "\t" << "kernel_size: 3" << "\n"
				<< "\t" << "\t" << "stride: 2" << "\n"
				<< "\t" << "\t" << "pad: 1" << "\n"
				<< "\t" << "\t" << "weight_filler {" << "\n"
				<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
				<< "\t" << "\t" << "}" << "\n"
				<< "\t" << "\t" << "bias_filler {" << "\n"
				<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
				<< "\t" << "\t" << "\t" << "value: 0" << "\n"
				<< "\t" << "\t" << "}" << "\n"
				<< "\t" << "}" << "\n"
				<< "}" << "\n"
				<< "\n";

			prototxt
				<< "layer {" << "\n"
				<< "\t" << "name: \"encoder_elu_" << current_size << "_3\"" << "\n"
				<< "\t" << "type: \"ELU\"" << "\n"
				<< "\t" << "bottom: \"encoder_size_" << current_size / 2 << "\"" << "\n"
				<< "\t" << "top: \"encoder_size_" << current_size / 2 << "\"" << "\n"
				<< "}" << "\n"
				<< "\n";
		}
		current_size /= 2;
		current_hidden_dim += hidden_dim;
	}

	current_size *= 2;

	//Last layer of the encoder to transform the output image in a vector
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"encoder_ip1\"" << "\n"
		<< "\t" << "type: \"InnerProduct\"" << "\n"
		<< "\t" << "bottom: \"encoder_conv_" << current_size << "_2\"" << "\n"
		<< "\t" << "top: \"encoded_h\"" << "\n"
		<< "\t" << "inner_product_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << h_dim << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Decoder starts here
	//Layers to transform the input vector into small images
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"decoder_ip1\"" << "\n"
		<< "\t" << "type: \"InnerProduct\"" << "\n"
		<< "\t" << "bottom: \"encoded_h\"" << "\n"
		<< "\t" << "top: \"decoder_ip1\"" << "\n"
		<< "\t" << "inner_product_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << hidden_dim * 8 * 8 << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"decoder_reshape_ip1\"" << "\n"
		<< "\t" << "type: \"Reshape\"" << "\n"
		<< "\t" << "bottom: \"decoder_ip1\"" << "\n"
		<< "\t" << "top: \"decoder_size_8\"" << "\n"
		<< "\t" << "reshape_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << hidden_dim << "\n"
		<< "\t" << "\t" << "\t" << "dim: 8" << "\n"
		<< "\t" << "\t" << "\t" << "dim: 8" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Add some convolution/upsampling until the dimension is the desired one
	while (current_size <= image_size)
	{
		//First convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"decoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"decoder_size_" << current_size << "\"" << "\n"
			<< "\t" << "top: \"decoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"decoder_elu_" << current_size << "_1\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"decoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"decoder_conv_" << current_size << "_1\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Second convolution + activation
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"decoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"Convolution\"" << "\n"
			<< "\t" << "bottom: \"decoder_conv_" << current_size << "_1\"" << "\n"
			<< "\t" << "top: \"decoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "convolution_param {" << "\n"
			<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
			<< "\t" << "\t" << "kernel_size: 3" << "\n"
			<< "\t" << "\t" << "stride: 1" << "\n"
			<< "\t" << "\t" << "pad: 1" << "\n"
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "bias_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: 0" << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"decoder_elu_" << current_size << "_2\"" << "\n"
			<< "\t" << "type: \"ELU\"" << "\n"
			<< "\t" << "bottom: \"decoder_conv_" << current_size << "_2\"" << "\n"
			<< "\t" << "top: \"decoder_conv_" << current_size << "_2\"" << "\n"
			<< "}" << "\n"
			<< "\n";

		//Upsampling
		if (current_size != image_size)
		{
			prototxt
				<< "layer {" << "\n"
				<< "\t" << "name: \"decoder_upsampling_" << current_size << "\"" << "\n"
				<< "\t" << "type: \"Deconvolution\"" << "\n"
				<< "\t" << "bottom: \"decoder_conv_" << current_size << "_2\"" << "\n"
				<< "\t" << "top: \"decoder_size_" << 2 * current_size << "\"" << "\n"
				<< "\t" << "convolution_param {" << "\n"
				<< "\t" << "\t" << "num_output: " << hidden_dim << "\n"
				<< "\t" << "\t" << "group: " << hidden_dim << "\n"
				<< "\t" << "\t" << "kernel_size: 2" << "\n"
				<< "\t" << "\t" << "stride: 2" << "\n"
				<< "\t" << "\t" << "pad: 0" << "\n"
				<< "\t" << "\t" << "weight_filler {" << "\n"
				<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
				<< "\t" << "\t" << "\t" << "value: 1" << "\n"
				<< "\t" << "\t" << "}" << "\n"
				<< "\t" << "\t" << "bias_term: false" << "\n"
				<< "\t" << "}" << "\n"
				<< "\t" << "param {" << "\n"
				<< "\t" << "\t" << "lr_mult: 0" << "\n"
				<< "\t" << "\t" << "decay_mult: 0" << "\n"
				<< "\t" << "}" << "\n"
				<< "}" << "\n"
				<< "\n";
		}
		current_size *= 2;
	}

	current_size /= 2;

	//Last layer to get a 3 channels image
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"decoder_last_conv\"" << "\n"
		<< "\t" << "type: \"Convolution\"" << "\n"
		<< "\t" << "bottom: \"decoder_conv_" << current_size << "_2\"" << "\n"
		<< "\t" << "top: \"decoded_image\"" << "\n"
		<< "\t" << "convolution_param {" << "\n"
		<< "\t" << "\t" << "num_output: 3" << "\n"
		<< "\t" << "\t" << "kernel_size: 3" << "\n"
		<< "\t" << "\t" << "stride: 1" << "\n"
		<< "\t" << "\t" << "pad: 1" << "\n"
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"xavier\"" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 0" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Create a L1-Loss with Caffe existing layers
	//Sum operation is done with two reduction layers
	//to avoid too small values after scale layer
	//(reduction could be done in one layer after scale)
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Loss_Sum_Layer\"" << "\n"
		<< "\t" << "type: \"Eltwise\"" << "\n"
		<< "\t" << "bottom: \"decoded_image\"" << "\n"
		<< "\t" << "bottom: \"image_input\"" << "\n"
		<< "\t" << "top: \"difference\"" << "\n"
		<< "\t" << "eltwise_param {" << "\n"
		<< "\t" << "\t" << "operation: SUM" << "\n"
		<< "\t" << "\t" << "coeff: 1" << "\n"
		<< "\t" << "\t" << "coeff: -1" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Loss_Reduction_Image\"" << "\n"
		<< "\t" << "type: \"Reduction\"" << "\n"
		<< "\t" << "bottom: \"difference\"" << "\n"
		<< "\t" << "top: \"summed_difference\"" << "\n"
		<< "\t" << "reduction_param {" << "\n"
		<< "\t" << "\t" << "operation: ASUM" << "\n"
		<< "\t" << "\t" << "axis: 1" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Loss_Scale\"" << "\n"
		<< "\t" << "type: \"Scale\"" << "\n"
		<< "\t" << "bottom: \"summed_difference\"" << "\n"
		<< "\t" << "top: \"scaled_difference\"" << "\n"
		<< "\t" << "scale_param {" << "\n"
		<< "\t" << "\t" << "filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: " << 1.0 / (double)(batch_size * 3 * image_size * image_size) << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "axis: 0" << "\n"
		<< "\t" << "\t" << "bias_term: false" << "\n"
		<< "\t" << "}" << "\n"
		<< "\t" << "param {" << "\n"
		<< "\t" << "\t" << "lr_mult: 0" << "\n"
		<< "\t" << "\t" << "decay_mult: 0" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Loss_Reduction_Batch\"" << "\n"
		<< "\t" << "type: \"Reduction\"" << "\n"
		<< "\t" << "bottom: \"scaled_difference\"" << "\n"
		<< "\t" << "top: \"L1_loss\"" << "\n"
		<< "\t" << "reduction_param {" << "\n"
		<< "\t" << "\t" << "operation: SUM" << "\n"
		<< "\t" << "}" << "\n"
		<< "\t" << "loss_weight: 1" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt.close();
}

int main(int argc, char** argv)
{
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	//In test mode we don't need all the caffe stuff
	if (!FLAGS_train)
	{
		for (int i = 0; i < google::NUM_SEVERITIES; ++i)
		{
			google::SetLogDestination(i, "");
		}
	}
	else
	{
		google::LogToStderr();
	}

	std::mt19937 random_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	CreateGeneratorPrototxt(FLAGS_batch_size, FLAGS_z_dim, FLAGS_hidden_dim, FLAGS_image_size);

	//Train
	if (FLAGS_train)
	{
		CreateDiscriminatorPrototxt(FLAGS_batch_size, FLAGS_h_dim, FLAGS_hidden_dim, FLAGS_image_size);

		NN_Agent nets(FLAGS_solver_gen, FLAGS_solver_discr, FLAGS_snapshot_generator, FLAGS_snapshot_discriminator, FLAGS_snapshot_k_t, FLAGS_weights_gen, FLAGS_weights_discr, "log.csv", 0.001f, 0.5f);

		std::cout << "Networks ready. Reading data ..." << std::endl;

		//Read data and store the file names
		std::vector<std::string > data_files_path;

		//Read image data (CelebA)
		boost::filesystem::path folder_data = FLAGS_training_dataset;

		if (boost::filesystem::is_empty(folder_data))
		{
			std::cerr << "Error, data folder not found or empty." << std::endl;
			return -1;
		}

		for (boost::filesystem::directory_entry &file : boost::filesystem::directory_iterator(folder_data))
		{
			if (boost::ends_with(file.path().string(), ".jpg") || boost::ends_with(file.path().string(), ".png") || boost::ends_with(file.path().string(), ".jpeg") || boost::ends_with(file.path().string(), ".bmp"))
			{
				data_files_path.push_back(file.path().string());
			}
		}

		std::vector<std::vector<float> > generator_input_test;

		if ((FLAGS_snapshot_generator.empty() && FLAGS_snapshot_discriminator.empty()) || FLAGS_preview_generator.empty())
		{
			std::string file_name = FLAGS_preview_generator.empty() ? "preview_values.csv" : FLAGS_preview_generator;
			std::ofstream preview_values_file(file_name);

			for (int i = 0; i < 16; ++i)
			{
				std::vector<float> current_gen_input;
				current_gen_input.reserve(FLAGS_z_dim);

				for (int j = 0; j < FLAGS_z_dim; ++j)
				{
					current_gen_input.push_back(std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_gen));
					preview_values_file << current_gen_input[j] << ";";
				}
				generator_input_test.push_back(current_gen_input);
				preview_values_file << std::endl;
			}
		}
		else
		{
			std::ifstream preview_values_file(FLAGS_preview_generator);
			float number = 0.0f;
			char character = 0;

			for (int i = 0; i < 16; ++i)
			{
				std::vector<float> current_gen_input;
				current_gen_input.reserve(FLAGS_z_dim);
				for (int j = 0; j < FLAGS_z_dim; ++j)
				{
					if (preview_values_file >> number >> character)
					{
						current_gen_input.push_back(number);
					}
					else
					{
						current_gen_input.push_back(std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_gen));
					}
				}
				generator_input_test.push_back(current_gen_input);
			}
		}

		int number_of_batch_in_epoch = data_files_path.size() / FLAGS_batch_size;

		//Alternate training on these two vectors for loading images while training
		std::vector<std::vector<float> > real_data_1;
		std::vector<std::vector<float> > real_data_2;

		//Data used for training or loading, alternatively real_data_1 and real_data_2
		std::vector<std::vector<float> > *used_data;
		std::vector<std::vector<float> > *loading_data;

		//Number of epochs
		for (int epoch = FLAGS_start_epoch; epoch < FLAGS_end_epoch; ++epoch)
		{
			//Shuffle the indices of the images
			std::vector<int> indices;
			indices.reserve(data_files_path.size());

			for (int i = 0; i < data_files_path.size(); ++i)
			{
				indices.push_back(i);
			}

			std::shuffle(indices.begin(), indices.end(), random_gen);

			//At the begining of an epoch, load a first part of the dataset into the first vector
			used_data = &real_data_2;
			loading_data = &real_data_1;
			std::thread thread_loading(LoadImagesFromFiles, data_files_path, std::vector<int>(indices.begin(), indices.begin() + 1 * FLAGS_number_batch_loaded * FLAGS_batch_size), loading_data, FLAGS_image_size);

			int index = 0;
			int fifth_of_epoch = 1;

			for (int batch = 0; batch < number_of_batch_in_epoch; ++batch)
			{
				//If there is a need to change the data vector
				if (batch % FLAGS_number_batch_loaded == 0)
				{
					thread_loading.join();
					used_data = (used_data == &real_data_1) ? &real_data_2 : &real_data_1;
					loading_data = (loading_data == &real_data_1) ? &real_data_2 : &real_data_1;
					int start_index = ((batch / FLAGS_number_batch_loaded) + 1) * FLAGS_number_batch_loaded * FLAGS_batch_size;
					int end_index = ((batch / FLAGS_number_batch_loaded) + 2) * FLAGS_number_batch_loaded * FLAGS_batch_size;
					end_index = std::min((unsigned long long)end_index, indices.size());
					if (start_index < end_index)
					{
						std::thread local_thread_loading(LoadImagesFromFiles, data_files_path, std::vector<int>(indices.begin() + start_index, indices.begin() + end_index), loading_data, FLAGS_image_size);
						thread_loading.swap(local_thread_loading);
					}
				}

				if (index > fifth_of_epoch * (indices.size() / 5) || index == 0)
				{
					if (index > 0)
					{
						fifth_of_epoch++;
					}
					std::vector<std::vector<float> > gen_output = nets.GeneratorForward(generator_input_test);

					//Reconstruct CelebA test images
					cv::Mat output = cv::Mat(FLAGS_image_size * 4, FLAGS_image_size * 4, CV_32FC3);
					for (int i = 0; i < 16; ++i)
					{
						cv::Mat current_output;

						std::vector<cv::Mat> rebuilt_channels;
						rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data()));
						rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data() + FLAGS_image_size * FLAGS_image_size));
						rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data() + 2 * FLAGS_image_size * FLAGS_image_size));

						cv::merge(rebuilt_channels, current_output);
						current_output *= 255.0f;

						current_output.copyTo(output(cv::Rect((i % 4) * FLAGS_image_size, (i / 4) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
					}
					std::string image_name = "Generation_epoch_" + std::to_string(epoch);

					if (index > 0)
					{
						image_name += "_iter_" + std::to_string(nets.Iter());
					}

					image_name += ".bmp";
					cv::imwrite(image_name, output);
				}

				std::vector<std::vector<float> > current_batch_data(used_data->begin() + (batch % FLAGS_number_batch_loaded) * FLAGS_batch_size, used_data->begin() + ((batch % FLAGS_number_batch_loaded) + 1) * FLAGS_batch_size);

				std::vector<std::vector<float> > current_batch_generator;

				for (int i = 0; i < current_batch_data.size(); ++i)
				{
					std::vector<float> current_gen_input;
					current_gen_input.reserve(FLAGS_z_dim);

					for (int j = 0; j < FLAGS_z_dim; ++j)
					{
						current_gen_input.push_back(std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_gen));
					}
					current_batch_generator.push_back(current_gen_input);
				}

				nets.Train(current_batch_generator, current_batch_data);

				index += FLAGS_batch_size;
			}
		}

		//After training, generate the faces once more
		std::vector<std::vector<float> > gen_output = nets.GeneratorForward(generator_input_test);

		cv::Mat output = cv::Mat(FLAGS_image_size * 4, FLAGS_image_size * 4, CV_32FC3);
		for (int i = 0; i < 16; ++i)
		{
			cv::Mat current_output;

			std::vector<cv::Mat> rebuilt_channels;
			rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data()));
			rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data() + FLAGS_image_size * FLAGS_image_size));
			rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, gen_output[i].data() + 2 * FLAGS_image_size * FLAGS_image_size));

			cv::merge(rebuilt_channels, current_output);
			current_output *= 255.0f;

			current_output.copyTo(output(cv::Rect((i % 4) * FLAGS_image_size, (i / 4) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
		}

		cv::imwrite("Fully_Trained.bmp", output);

		return 0;
	}
	//Test
	else
	{
		NN_Agent net("generator.prototxt", FLAGS_weights_gen);

		int index = 0;
		while (index != FLAGS_num_gen)
		{
			cv::Mat interpolation(FLAGS_image_size * 10, FLAGS_image_size * 10, CV_32FC3);

			for (int i = 0; i < 10; ++i)
			{
				//Create the inputs of the generator
				std::vector<std::vector<float> > inter_faces_input(10, std::vector<float>(FLAGS_z_dim, 0.0f));
				for (int j = 0; j < FLAGS_z_dim; ++j)
				{
					float val1 = std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_gen);
					float val2 = std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_gen);

					for (int k = 0; k < 10; ++k)
					{
						inter_faces_input[k][j] = (1.0f - (float)k / 9.0f) * val1 + (float)k / 9.0f * val2;
					}
				}

				//Get output
				std::vector<std::vector<float> > inter_faces = net.GeneratorForward(inter_faces_input);

				//Convert output to images and copy them to display image
				for (int j = 0; j < inter_faces_input.size(); ++j)
				{
					cv::Mat current_output;

					std::vector<cv::Mat> rebuilt_channels;
					rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, inter_faces[j].data()));
					rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, inter_faces[j].data() + FLAGS_image_size * FLAGS_image_size));
					rebuilt_channels.push_back(cv::Mat(FLAGS_image_size, FLAGS_image_size, CV_32FC1, inter_faces[j].data() + 2 * FLAGS_image_size * FLAGS_image_size));

					cv::merge(rebuilt_channels, current_output);
					current_output *= 255.0f;

					current_output.copyTo(interpolation(cv::Rect(j * FLAGS_image_size, i * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
				}
			}

			if (FLAGS_save_img)
			{
				cv::imwrite("Generated_" + std::to_string(index) + ".bmp", interpolation);
			}

			interpolation /= 255.0f;
			cv::resize(interpolation, interpolation, cv::Size(600, 600));
			cv::imshow("Interpolation", interpolation);
			cv::waitKey(0);
			index++;
		}

		return 0;
	}
}