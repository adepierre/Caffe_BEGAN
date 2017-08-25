#pragma once

#include <string>
#include <vector>

#include <caffe/caffe.hpp>


class NN_Agent
{
public:
	/**
	* \brief Create an agent for training, initialize net and solver
	* \param solver_generator_ Caffe solver for generator network (*.prototxt file)
	* \param solver_discirminator_ Caffe solver for discriminator network (*.prototxt file)
	* \param snapshot_generator Caffe snapshot file to resume training of the generator (*.solverstate file)
	* \param snapshot_discriminator Caffe snapshot file to resume training of the discriminator (*.solverstate file)
	* \param snapshot_k_t k_t file saved during training
	* \param weights_generator Caffe weights file to fine tune generator (*.caffemodel)
	* \param weights_discriminator Caffe weights file to fine tune discriminator (*.caffemodel)
	* \param log_filename File in which the logs will be written
	* \param lambda_k_ k_t learning rate
	* \param gamma_ Diversity ratio
	*/
	NN_Agent(const std::string &solver_generator_,
			 const std::string &solver_discriminator_,
			 const std::string &snapshot_generator,
			 const std::string &snapshot_discriminator,
			 const std::string &snapshot_k_t,
			 const std::string &weights_generator,
			 const std::string &weights_discriminator,
			 const std::string &log_filename,
			 const float &lambda_k_,
			 const float &gamma_);

	/**
	* \brief Create an agent for testing, initialize the net
	* \param model_file Caffe model file for generator net (*.prototxt)
	* \param trained_file Caffe caffemodel file to fill weights (*.caffemodel)
	*/
	NN_Agent(const std::string &model_file,
			 const std::string &trained_file);

	~NN_Agent();

	/**
	* \brief Perform one train cycle on the whole net (generator+discriminator)
	* \param generator_input One batch of input for the generator
	* \param true_data One batch of true images for the discriminator
	*/
	void Train(const std::vector<std::vector<float> > &generator_input, const std::vector<std::vector<float> > &true_data);

	/**
	* \brief Perform one forward pass of the generator
	* \param generator_input One batch of input for the generator
	* \return One batch of generated data
	*/
	std::vector<std::vector<float> > GeneratorForward(const std::vector<std::vector<float> > &generator_input);

	/**
	* \brief Get current solver iteration
	* \return Current solver iteration
	*/
	int Iter();

protected:
	/**
	* \brief Fill the input blob of the generator with data
	* \param generator_input One batch of input for the generator
	*/
	void SetInputGenerator(const std::vector<std::vector<float> > &generator_input);

	/**
	* \brief Fill the input blob of the generator with data
	* \param discriminator_input One batch of input for the discriminator
	*/
	void SetInputDiscriminator(const std::vector<std::vector<float> > &discriminator_input);

	/**
	* \brief Perform one forward pass on the discriminator
	* \param discriminator_input One batch of input for the discriminator
	* \return The loss returned by the forward pass
	*/
	float DiscriminatorForward(const std::vector<std::vector<float> > &discriminator_input);
	
protected:
	//Common parameters
	boost::shared_ptr<caffe::Net<float> > net_generator;

	boost::shared_ptr<caffe::Blob<float> > input_generator;
	boost::shared_ptr<caffe::Blob<float> > output_generator;
		
	//Parameters used for training
	boost::shared_ptr<caffe::Solver<float> > solver_discriminator;
	boost::shared_ptr<caffe::Net<float> > net_discriminator;
	boost::shared_ptr<caffe::Solver<float> > solver_generator;

	boost::shared_ptr<caffe::Blob<float> > input_discriminator;
	boost::shared_ptr<caffe::Blob<float> > regenerated_discriminator;
	boost::shared_ptr<caffe::Blob<float> > loss_discriminator;

	float k_t;
	float lambda_k;
	float gamma;

	std::ofstream log_file;

	int display_interval;

	float mean_convergence;
	float mean_real_loss;
	float mean_generated_loss;
	float mean_kt;
};


