#include "NN_Agent.h"

NN_Agent::NN_Agent(const std::string &solver_generator_, 
				   const std::string &solver_discriminator_,
				   const std::string &snapshot_generator, 
				   const std::string &snapshot_discriminator, 
				   const std::string &snapshot_k_t,
				   const std::string &weights_generator, 
				   const std::string &weights_discriminator,
				   const std::string &log_filename,
				   const float &lambda_k_,
				   const float &gamma_):
		lambda_k(lambda_k_),
		gamma(gamma_)
{
	k_t = 0.0f;

	mean_convergence = 0.0f;
	mean_generated_loss = 0.0f;
	mean_real_loss = 0.0f;
	mean_kt = 0.0f;

	//Create caffe objects (solver + net)
	caffe::SolverParameter solver_param_generator, solver_param_discriminator;
	caffe::ReadProtoFromTextFileOrDie(solver_generator_, &solver_param_generator);
	caffe::ReadProtoFromTextFileOrDie(solver_discriminator_, &solver_param_discriminator);

	solver_generator.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_generator));
	solver_discriminator.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_discriminator));
	net_generator = solver_generator->net();
	net_discriminator = solver_discriminator->net();

	if (snapshot_generator.empty())
	{
		std::cout << "Starting new training for generator" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_generator << " for generator net." << std::endl;
		solver_generator->Restore(snapshot_generator.c_str());
	}

	if (snapshot_discriminator.empty())
	{
		std::cout << "Starting new training for discriminator" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_discriminator << " for discriminator net." << std::endl;
		solver_discriminator->Restore(snapshot_discriminator.c_str());
	}

	if (!snapshot_k_t.empty())
	{
		std::ifstream k_t_file(snapshot_k_t);
		k_t_file >> k_t;
		k_t_file.close();
	}

	if (!weights_generator.empty())
	{
		std::cout << "Copying generator weights from ... " << weights_generator << std::endl;
		net_generator->CopyTrainedLayersFrom(weights_generator);
	}

	if (!weights_discriminator.empty())
	{
		std::cout << "Copying discriminator weights from ... " << weights_discriminator << std::endl;
		net_discriminator->CopyTrainedLayersFrom(weights_discriminator);
	}

	//Get input and output blobs
	input_generator = net_generator->blob_by_name("generator_z");
	output_generator = net_generator->blob_by_name("generated_image");

	input_discriminator = net_discriminator->blob_by_name("image_input");
	regenerated_discriminator = net_discriminator->blob_by_name("decoded_image");
	loss_discriminator = net_discriminator->blob_by_name("L1_loss");

	display_interval = solver_discriminator->param().display();

	if (solver_discriminator->iter() > 0 || solver_generator->iter() > 0)
	{
		log_file.open(log_filename, std::ofstream::out|std::ofstream::app);
	}
	else
	{
		log_file.open(log_filename, std::ofstream::out);
		log_file << "Iter;k_t;Convergence;Real loss;Generated loss" << std::endl;
	}
}

NN_Agent::NN_Agent(const std::string &model_file, 
				   const std::string &trained_file)
{
	net_generator.reset(new caffe::Net<float>(model_file, caffe::TEST));

	if (!trained_file.empty())
	{
		net_generator->CopyTrainedLayersFrom(trained_file);
	}

	input_generator = net_generator->blob_by_name("generator_z");
	output_generator = net_generator->blob_by_name("generated_image");
}

NN_Agent::~NN_Agent()
{
}

void NN_Agent::Train(const std::vector<std::vector<float> > &generator_input, const std::vector<std::vector<float> > &true_data)
{
	net_discriminator->ClearParamDiffs();
	net_generator->ClearParamDiffs();

	//Generate images with the generator
	std::vector<std::vector<float> > generated_data = GeneratorForward(generator_input);

	//Compute loss of generated images
	float generated_loss = DiscriminatorForward(generated_data);
	
	//Compute gradient w.r.t. the generator and train generator with it
	net_discriminator->Backward();

	caffe::caffe_cpu_scale(input_discriminator->count(), 1.0f, input_discriminator->cpu_diff(), output_generator->mutable_cpu_diff());

	net_generator->Backward();

	solver_generator->ApplyUpdate();
	solver_generator->iter_++;

	net_discriminator->ClearParamDiffs();

	//Add the generator loss part to the diffs -k_t*L(G(z))
	if (k_t != 0.0f)
	{
		net_discriminator->BackwardFromTo(net_discriminator->layers().size() - 1, net_discriminator->layers().size() - 4);
		caffe::caffe_scal(regenerated_discriminator->count(), -k_t, regenerated_discriminator->mutable_cpu_diff());
		net_discriminator->BackwardFrom(net_discriminator->layers().size() - 5);
	}
	
	//Run the discriminator on real data
	float real_loss = DiscriminatorForward(true_data);

	//Backprop to add the real loss to the diffs L(x))
	net_discriminator->Backward();

	solver_discriminator->ApplyUpdate();
	solver_discriminator->iter_++;	

	float convergence = real_loss + abs(gamma * real_loss - generated_loss);

	mean_convergence += convergence;
	mean_real_loss += real_loss;
	mean_generated_loss += generated_loss;
	mean_kt += k_t;

	//Snapshot and display
	if ((solver_discriminator->iter() % solver_discriminator->param().snapshot() == 0) && solver_discriminator->iter() > 0)
	{
		solver_discriminator->Snapshot();
		std::ofstream k_t_file("k_t_iter_" + std::to_string(solver_discriminator->iter()) + ".txt");
		k_t_file << k_t;
		k_t_file.close();
	}

	if ((solver_generator->iter() % solver_generator->param().snapshot() == 0) && solver_generator->iter() > 0)
	{
		solver_generator->Snapshot();		
		std::ofstream k_t_file("k_t_iter_" + std::to_string(solver_generator->iter()) + ".txt");
		k_t_file << k_t;
		k_t_file.close();
	}

	//Update k_t
	k_t = std::max(std::min(k_t + lambda_k * (gamma * real_loss - generated_loss), 1.0f), 0.0f);

	if ((solver_discriminator->iter() % display_interval == 0))
	{
		std::cout << "Convergence value: " << convergence << std::endl;
		if (log_file.is_open())
		{
			log_file << solver_discriminator->iter() << ";" << mean_kt / display_interval << ";" << mean_convergence / display_interval << ";" << mean_real_loss / display_interval << ";" << mean_generated_loss / display_interval << std::endl;
		}
		mean_convergence = 0.0f;
		mean_generated_loss = 0.0f;
		mean_real_loss = 0.0f;
		mean_kt = 0.0f;
	}

}

std::vector<std::vector<float> > NN_Agent::GeneratorForward(const std::vector<std::vector<float> > &generator_input)
{
	SetInputGenerator(generator_input);

	net_generator->Forward();

	std::vector<std::vector<float> > output(output_generator->shape()[0], std::vector<float>(output_generator->count() / output_generator->shape()[0], 0.0f));
	int offset = 0;

	for (size_t i = 0; i < output.size(); ++i)
	{
		for (size_t j = 0; j < output[i].size(); ++j)
		{
			output[i][j] = output_generator->cpu_data()[offset];
			offset++;
		}
	}

	return output;
}

int NN_Agent::Iter()
{
	return solver_discriminator->iter();
}

void NN_Agent::SetInputGenerator(const std::vector<std::vector<float> > &generator_input)
{
	int offset = 0;

	for (size_t i = 0; i < generator_input.size(); ++i)
	{
		for (size_t j = 0; j < generator_input[i].size(); ++j)
		{
			input_generator->mutable_cpu_data()[offset] = generator_input[i][j];
			offset++;
		}
	}
}

void NN_Agent::SetInputDiscriminator(const std::vector<std::vector<float> > &discriminator_input)
{
	int offset = 0;

	for (size_t i = 0; i < discriminator_input.size(); ++i)
	{
		for (size_t j = 0; j < discriminator_input[i].size(); ++j)
		{
			input_discriminator->mutable_cpu_data()[offset] = discriminator_input[i][j];
			offset++;
		}
	}
}

float NN_Agent::DiscriminatorForward(const std::vector<std::vector<float>>& discriminator_input)
{
	SetInputDiscriminator(discriminator_input);

	net_discriminator->Forward();

	float loss = loss_discriminator->cpu_data()[0];
	
	return loss;
}