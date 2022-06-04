/**
 *
 * APPFIS version of Triton
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef TRITON_APPFIS_HPP
#define TRITON_APPFIS_HPP


#include <iostream>
#include "../../Appfis/Appfis.hpp"
#include "inflow.hpp"
#include "config_utils.hpp"
#include <cfloat>

using namespace std;

namespace APPFIS
{
	PARALLEL void STENCIL(const int i, const int j, double dx, double cn, double hextra,
		Array<double>* input_qx, Array<double>* input_qy, Array<double>* input_h, Array<double>* output);

	PARALLEL void STENCIL(const int i, const int j, Array<double>* h_arr, Array<double>* sqrth,
		Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1);

	PARALLEL void STENCIL(const int ii, const int jj, double dx, double dt, double hextra,
		Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* sqrth,
		Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1);

	PARALLEL void STENCIL(const int ii, const int jj, double dx, double dt, double hextra, int dummy,
		Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* sqrth,
		Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1);

	PARALLEL void STENCIL(const int i, const int j, double dt, double hextra,
		Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* n_arr,
		Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1);

	PARALLEL void STENCIL(const int i, const int j, double dt, double hextra,
		Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem);

	PARALLEL void STENCIL(const int i, int size, double cellsize, double global_dt, double simtime, int idx_low, int idx_high,
		Array<int>* src_g, Array<double>* src_time, Array<double>* src_val, Array<double>* h_arr);
}

void Triton(int argc, char* argv[]);
void load_header_from_dem_file_ascii(std::string filename);
void load_header_from_dem_file_binary(std::string filename);
int calc_src_col(double src_x, double xllc, double cell_size_);
int calc_src_row(double src_y, double yllc, double cell_size_, int nrows);


#define G 9.81
#define SQRTG 3.132091953
#define EPS12 1e-12

int dimX, dimY, no_data_value;
double xllcorner, yllcorner, cellsize;


void Triton(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int thread = 1;
	bool overlap = false;
	bool outputFlag = false;

	if (APPFIS::CheckArgument(argc, argv, "thread")) {
		thread = APPFIS::GetArgument(argc, argv, "thread");
	}
	overlap = APPFIS::CheckArgument(argc, argv, "overlap");
	outputFlag = APPFIS::CheckArgument(argc, argv, "output");

	std::string cfg_dir = std::string(argv[1]);

	APPFIS::Attribute attribute;
	attribute.threads = thread;
	attribute.overlap = overlap;
	attribute.ghostLayer = 1;
	attribute.periodic = false;

	APPFIS::Initialize(argc, argv, APPFIS::DIM_2D, APPFIS::DIM_2D, attribute);

	std::string cfg_content = file_content_to_string(cfg_dir);
	arguments<double> arglist = get_args<double>(cfg_content);
	double simtime = arglist.sim_start_time;

	hydrograph<double> hyg;
	if (arglist.num_sources > 0)
	{
		hyg = hydrograph<double>(arglist.hydrograph_filename);
		hyg.convert_time_hr_to_secs();
	}

	if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
	{
		load_header_from_dem_file_ascii(arglist.dem_filename);
	}
	else
	{
		load_header_from_dem_file_binary(arglist.dem_filename);
	}

	APPFIS::Grid<double> dem(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> nin(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> hin(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> uin(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> vin(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);

	APPFIS::Grid<double> sqrth(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsh0(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsh1(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsqx0(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsqx1(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsqy0(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	APPFIS::Grid<double> rhsqy1(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);

	APPFIS::Grid<double> dt_values_arr(APPFIS::DIM_2D, dimX, dimY, APPFIS::DUAL);

	if (APPFIS::IsMaster())
	{
		dem.AllocateFullGrid();
		nin.AllocateFullGrid();
		hin.AllocateFullGrid();
		uin.AllocateFullGrid();
		vin.AllocateFullGrid();

		if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
		{
			dem.GetFullGrid()->LoadAsciiFile(arglist.dem_filename, 6);
		}
		else
		{
			dem.GetFullGrid()->LoadBinaryFile(arglist.dem_filename, 6);
		}

		if (!arglist.n_infile.empty())
		{
			if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
			{
				nin.GetFullGrid()->LoadAsciiFile(arglist.n_infile);
			}
			else
			{
				nin.GetFullGrid()->LoadBinaryFile(arglist.n_infile);
			}
		}
		else
		{
			nin.GetFullGrid()->FillData(arglist.const_mann);
		}
		nin.GetFullGrid()->Square();

		if (arglist.h_infile.size() > 0)
		{
			if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
			{
				hin.GetFullGrid()->LoadAsciiFile(arglist.h_infile);
			}
			else
			{
				hin.GetFullGrid()->LoadBinaryFile(arglist.h_infile);
			}
		}

		if (arglist.qx_infile.size() > 0)
		{
			if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
			{
				uin.GetFullGrid()->LoadAsciiFile(arglist.qx_infile);
			}
			else
			{
				uin.GetFullGrid()->LoadBinaryFile(arglist.qx_infile);
			}
		}
		if (arglist.qy_infile.size() > 0)
		{
			if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
			{
				vin.GetFullGrid()->LoadAsciiFile(arglist.qy_infile);
			}
			else
			{
				vin.GetFullGrid()->LoadBinaryFile(arglist.qy_infile);
			}
		}
	}

	APPFIS::Scatter(dem.GetFullGrid(), dem.GetSubGrid());
	dem.SetBoundary(1e6);

	APPFIS::Scatter(nin.GetFullGrid(), nin.GetSubGrid());
	nin.SetBoundaryCopy();

	if (arglist.h_infile.size() > 0)
	{
		APPFIS::Scatter(hin.GetFullGrid(), hin.GetSubGrid());
	}

	if (arglist.qx_infile.size() > 0)
	{
		APPFIS::Scatter(uin.GetFullGrid(), uin.GetSubGrid());
	}

	if (arglist.qy_infile.size() > 0)
	{
		APPFIS::Scatter(vin.GetFullGrid(), vin.GetSubGrid());
	}


	APPFIS::Grid<int> src_grid(APPFIS::DIM_2D, dimX, dimY, APPFIS::SINGLE);
	if (APPFIS::IsMaster())
	{
		src_grid.AllocateFullGrid();
		if (arglist.num_sources > 0)
		{
			APPFIS::Array<int>* src_arr = src_grid.GetFullGrid();

			std::vector<double> src_x = arglist.src_x_loc;
			std::vector<double> src_y = arglist.src_y_loc;

			int num_sources = arglist.num_sources;

			for (int i = 0; i < num_sources; ++i)
			{
				int src_cols_index = calc_src_col(src_x[i], xllcorner, cellsize);
				int src_rows_index = calc_src_row(src_y[i], yllcorner, cellsize, dimY);
				int src_index = src_rows_index * dimX + src_cols_index;

				src_arr->Set(src_index, i + 1);
			}
		}
	}

	APPFIS::Scatter(src_grid.GetFullGrid(), src_grid.GetSubGrid());

	int no_of_src = 0;
	vector<int> relative_src_index;
	vector<int> source_cells;

	APPFIS::Array<int>* src_arr = src_grid.GetSubGrid();
	for (int i = 0; i < src_arr->GetSize(); i++)
	{
		if (src_arr->Get(i) > 0)
		{
			no_of_src++;
			relative_src_index.push_back(src_arr->Get(i));
			source_cells.push_back(i);
		}
	}

	int src_grid_final_size = 1;
	if (no_of_src > 0)
	{
		src_grid_final_size = no_of_src;
	}
	APPFIS::Grid<int> src_grid_final(APPFIS::DIM_1D, src_grid_final_size, APPFIS::FLAT);

	for (int i = 0; i < no_of_src; i++)
	{
		src_grid_final.GetSubGrid()->Set(i, source_cells[i]);
	}

	int src_time_arr_size = 1;
	if (no_of_src > 0)
	{
		src_time_arr_size = hyg.get_num_inflow_rows();
	}
	APPFIS::Grid<double> src_time_arr(APPFIS::DIM_1D, src_time_arr_size, APPFIS::FLAT);

	int src_val_arr_size = 1;
	if (no_of_src > 0)
	{
		src_val_arr_size = src_grid_final_size * src_time_arr_size;
	}
	APPFIS::Grid<double> src_val_arr(APPFIS::DIM_1D, src_val_arr_size, APPFIS::FLAT);

	if (no_of_src > 0)
	{
		for (int i = 0; i < src_time_arr_size; i++)
		{
			src_time_arr.GetSubGrid()->Set(i, hyg.get_time_at(i));

			for (int j = 0; j < no_of_src; j++)
			{
				int src_number = relative_src_index[j];
				src_val_arr.GetSubGrid()->Set(i * no_of_src + j, hyg.get_flow_at(i, src_number));
			}
		}
	}

	APPFIS::Config<double> config(dem);
	APPFIS::Config<double> configF(dem, 0, 0, 0, 0);
	APPFIS::Config<double> configX(dem, 0, 1, 1, 1);
	APPFIS::Config<double> configY(dem, 1, 1, 1, 0);

	APPFIS::Config<double> configS(no_of_src, 0, 0);
	APPFIS::Config<double> configQ(dem);
	configQ.GridsToUpdate(uin, vin);

	APPFIS::InitializeExecution(dem, nin, hin, uin, vin, sqrth, rhsh0, rhsh1, rhsqx0, rhsqx1, rhsqy0, rhsqy1, dt_values_arr);
	APPFIS::InitializeExecution(src_grid_final);
	APPFIS::InitializeExecution(src_time_arr, src_val_arr);

	double global_dt = arglist.time_step;
	int it_count = arglist.it_count;
	int print_id = arglist.checkpoint_id;
	int idx_low = 0;

	timer.Start("COMPUTE");
	while (simtime < arglist.sim_duration)
	{
		it_count++;

		if (!arglist.time_increment_fixed)
		{
			APPFIS::Execute2D(config, cellsize, arglist.courant, arglist.hextra,
				uin.GetSubGrid(), vin.GetSubGrid(), hin.GetSubGrid(), dt_values_arr.GetSubGrid());

			APPFIS::Execute2DReduce(config, global_dt, APPFIS::MIN, dt_values_arr);
			if (global_dt >= DBL_MAX - 1.0)
			{
				global_dt = arglist.time_step;
			}

			if (simtime + global_dt > arglist.print_interval * (print_id + 1))
			{
				global_dt = arglist.print_interval * (print_id + 1) - simtime;
			}
		}

		APPFIS::Execute2D(configF, hin.GetSubGrid(), sqrth.GetSubGrid(),
			rhsh0.GetSubGrid(), rhsh1.GetSubGrid(), rhsqx0.GetSubGrid(), rhsqx1.GetSubGrid(), rhsqy0.GetSubGrid(), rhsqy1.GetSubGrid());

		APPFIS::Execute2D(configX, cellsize, global_dt, arglist.hextra,
			hin.GetSubGrid(), uin.GetSubGrid(), vin.GetSubGrid(), dem.GetSubGrid(), sqrth.GetSubGrid(),
			rhsh0.GetSubGrid(), rhsh1.GetSubGrid(), rhsqx0.GetSubGrid(), rhsqx1.GetSubGrid(), rhsqy0.GetSubGrid(), rhsqy1.GetSubGrid());

		APPFIS::Execute2D(configY, cellsize, global_dt, arglist.hextra, 0,
			hin.GetSubGrid(), uin.GetSubGrid(), vin.GetSubGrid(), dem.GetSubGrid(), sqrth.GetSubGrid(),
			rhsh0.GetSubGrid(), rhsh1.GetSubGrid(), rhsqx0.GetSubGrid(), rhsqx1.GetSubGrid(), rhsqy0.GetSubGrid(), rhsqy1.GetSubGrid());

		APPFIS::Execute2D(config, global_dt, arglist.hextra,
			hin.GetSubGrid(), uin.GetSubGrid(), vin.GetSubGrid(), dem.GetSubGrid(), nin.GetSubGrid(),
			rhsh0.GetSubGrid(), rhsh1.GetSubGrid(), rhsqx0.GetSubGrid(), rhsqx1.GetSubGrid(), rhsqy0.GetSubGrid(), rhsqy1.GetSubGrid());

		if (no_of_src > 0)
		{
			bool check_flag = false;
			int idx_high = hyg.get_num_inflow_rows() - 1;
			for (int i = idx_low; i < idx_high; i++)
			{
				if (hyg.get_time_at(i + 1) > simtime)
				{
					idx_low = i;
					idx_high = i + 1;
					check_flag = true;
					break;
				}
			}
			if (!check_flag)
			{
				idx_low = idx_high;
			}

			APPFIS::Execute1D(configS, no_of_src, cellsize, global_dt, simtime, idx_low, idx_high,
				src_grid_final.GetSubGrid(), src_time_arr.GetSubGrid(), src_val_arr.GetSubGrid(), hin.GetSubGrid());
		}

		APPFIS::HaloExchange(&hin);

		APPFIS::Execute2D(configQ, global_dt, arglist.hextra,
			hin.GetSubGrid(), uin.GetSubGrid(), vin.GetSubGrid(), dem.GetSubGrid());

		simtime += global_dt;

		if (simtime >= arglist.print_interval * (print_id + 1))
		{
			print_id++;
		}
	}
	timer.Stop("COMPUTE");

	APPFIS::FinalizeExecution(hin, uin, vin);
	APPFIS::Gather(hin.GetFullGrid(), hin.GetSubGrid());
	APPFIS::Gather(uin.GetFullGrid(), uin.GetSubGrid());
	APPFIS::Gather(vin.GetFullGrid(), vin.GetSubGrid());

	timer.Stop("TOTAL");
	if (APPFIS::IsMaster())
	{
		std::cout << "Number of iteration: " << it_count << std::endl;

		double totalTime = timer.GetCustomTime("TOTAL");
		double computeTime = timer.GetCustomTime("COMPUTE");
		std::cout << "Total: " << totalTime << std::endl;
		std::cout << "Compute: " << computeTime << std::endl;
		std::cout << "BLUPS: " << timer.CalculateBLUPS("COMPUTE", dimX * dimY, it_count) << std::endl;

		if (outputFlag)
		{
			std::string fileNameh("./output/h.txt");
			hin.GetFullGrid()->SaveAsciiFile(fileNameh);

			std::string fileNameu("./output/u.txt");
			uin.GetFullGrid()->SaveAsciiFile(fileNameu);

			std::string fileNamev("./output/v.txt");
			vin.GetFullGrid()->SaveAsciiFile(fileNamev);
		}
	}

	APPFIS::Finalize();
}


void load_header_from_dem_file_ascii(std::string filename)
{
	std::ifstream ifs(filename.c_str());
	if (!ifs.good())
	{
		std::cerr << "Error reading file: " << filename << std::endl;
		exit(EXIT_FAILURE);

	}

	int line_num = 0;

	for (;;)
	{
		std::string line;
		std::getline(ifs, line);
		if (!ifs)
			break;

		line_num++;
		if (line_num > 6)
			break;

		line = APPFIS::Trim(line);
		std::vector<std::string> tokens = APPFIS::Split(line, ' ');
		const char* value = (*(tokens.end() - 1)).c_str();

		if (!line.empty() && line[0] != '#')
		{
			switch (line_num)
			{
			case 1:
			{
				dimX = atoi(value);
				break;
			}
			case 2:
			{
				dimY = atoi(value);
				break;
			}
			case 3:
			{
				xllcorner = atof(value);
				break;
			}
			case 4:
			{
				yllcorner = atof(value);
				break;
			}
			case 5:
			{
				cellsize = atof(value);
				break;
			}
			case 6:
			{
				no_data_value = atoi(value);
				break;
			}
			default:
			{

			}
			}
		}
	}
	ifs.close();
}


void load_header_from_dem_file_binary(std::string filename)
{
	std::ifstream ifs(filename.c_str(), ios::binary);
	if (!ifs.good())
	{
		std::cerr << "Error reading file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	double* arr = new double[6];

	ifs.read((char*)arr, sizeof(double) * 6);

	ifs.close();

	for (int i = 0; i < 6; i++)
	{
		int line_num = i + 1;;
		double value = arr[i];

		switch (line_num)
		{
		case 1:
		{
			dimX = (int)(value);
			break;
		}
		case 2:
		{
			dimY = (int)(value);
			break;
		}
		case 3:
		{
			xllcorner = value;
			break;
		}
		case 4:
		{
			yllcorner = value;
			break;
		}
		case 5:
		{
			cellsize = value;
			break;
		}
		case 6:
		{
			no_data_value = (int)(value);;
			break;
		}
		default:
		{

		}
		}
	}
}


int calc_src_col(double src_x, double xllc, double cell_size_)
{
	return ceil(((src_x - xllc) / cell_size_)) - 1;
}


int calc_src_row(double src_y, double yllc, double cell_size_, int nrows)
{
	return ceil((nrows - ((src_y - yllc) / cell_size_))) - 1;
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, double dx, double cn, double hextra,
	Array<double>* input_qx, Array<double>* input_qy, Array<double>* input_h, Array<double>* output)
{
	output->Set(i, j, DBL_MAX);
	double hij = input_h->Get(i, j);

	if (hij > hextra)
	{
		double dtx = (dx / (fabs(input_qx->Get(i, j) / hij) + sqrt(G * hij)));
		double dty = (dx / (fabs(input_qy->Get(i, j) / hij) + sqrt(G * hij)));
		if (dtx < dty)
		{
			output->Set(i, j, cn * dtx);
		}
		else
		{
			output->Set(i, j, cn * dty);
		}
	}
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, Array<double>* h_arr, Array<double>* sqrth,
	Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1)
{
	double hij = h_arr->Get(i, j);
	double sqrthij = 0.0;
	if (hij > 0.0)
	{
		sqrthij = sqrt(hij);
	}
	sqrth->Set(i, j, sqrthij);
	rhsh0->Set(i, j, 0.0);
	rhsh1->Set(i, j, 0.0);
	rhsqx0->Set(i, j, 0.0);
	rhsqx1->Set(i, j, 0.0);
	rhsqy0->Set(i, j, 0.0);
	rhsqy1->Set(i, j, 0.0);
}


PARALLEL void APPFIS::STENCIL(const int ii, const int jj, double dx, double dt, double hextra,
	Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* sqrth,
	Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1)
{
	int id1, id2;
	double nx, ny;

	id1 = ii;
	id2 = ii + 1;
	nx = 1.0;
	ny = 0.0;

	double h1 = h_arr->Get(id1, jj);
	double h2 = h_arr->Get(id2, jj);

	if (h1 > 0.0 || h2 > 0.0)
	{
		int i;
		double alpha[3], beta[3], eigen[3], eigenE[3], eigev[3][3], d1[3], d2[3];
		double u1, u2, v1, v2;

		double qx1 = qx_arr->Get(id1, jj);
		double qy1 = qy_arr->Get(id1, jj);
		double z1 = dem->Get(id1, jj);

		double qx2 = qx_arr->Get(id2, jj);
		double qy2 = qy_arr->Get(id2, jj);
		double z2 = dem->Get(id2, jj);

		if (h1 >= hextra)
		{
			u1 = qx1 / h1;
			v1 = qy1 / h1;
		}
		else
		{
			u1 = 0.0;
			v1 = 0.0;
		}
		if (h2 >= hextra)
		{
			u2 = qx2 / h2;
			v2 = qy2 / h2;
		}
		else
		{
			u2 = 0.0;
			v2 = 0.0;
		}

		double sqh1 = sqrth->Get(id1, jj);
		double sqh2 = sqrth->Get(id2, jj);

		double h12 = 0.5 * (h1 + h2);
		double dh = h2 - h1;
		double c12 = sqrt(G * h12);

		double u12 = (u1 * sqh1 + u2 * sqh2) / (sqh1 + sqh2);
		double v12 = (v1 * sqh1 + v2 * sqh2) / (sqh1 + sqh2);

		double un = u12 * nx + v12 * ny;

		eigen[0] = un - c12;
		eigen[1] = un;
		eigen[2] = un + c12;

		//entropy correction
		for (i = 0; i < 3; i++)
		{
			eigenE[i] = 0.0;
		}

		double e1 = u1 * nx + v1 * ny - SQRTG * sqh1;
		double e2 = u2 * nx + v2 * ny - SQRTG * sqh2;
		if (e1 < 0.0 && e2>0.0)
		{
			eigenE[0] = eigen[0] - e1 * (e2 - eigen[0]) / (e2 - e1);
			eigen[0] = e1 * (e2 - eigen[0]) / (e2 - e1);
		}

		e1 = u1 * nx + v1 * ny + SQRTG * sqh1;
		e2 = u2 * nx + v2 * ny + SQRTG * sqh2;

		if (e1 < 0.0 && e2>0.0)
		{
			eigenE[2] = eigen[2] - e2 * (eigen[2] - e1) / (e2 - e1);
			eigen[2] = e2 * (eigen[2] - e1) / (e2 - e1);
		}

		eigev[0][0] = 1.0;
		eigev[0][1] = u12 - c12 * nx;
		eigev[0][2] = v12 - c12 * ny;
		eigev[1][0] = 0.0;
		eigev[1][1] = -c12 * ny;
		eigev[1][2] = c12 * nx;
		eigev[2][0] = 1.0;
		eigev[2][1] = u12 + c12 * nx;
		eigev[2][2] = v12 + c12 * ny;

		alpha[0] = 0.5 * (dh - (((qx2 - qx1) * nx + (qy2 - qy1) * ny) - un * dh) / c12);
		alpha[1] = (((qy2 - qy1) - v12 * dh) * nx - ((qx2 - qx1) - u12 * dh) * ny) / c12;
		alpha[2] = 0.5 * (dh + (((qx2 - qx1) * nx + (qy2 - qy1) * ny) - un * dh) / c12);

		double dz = z2 - z1;
		double l1 = z1 + h1;
		double l2 = z2 + h2;
		double dzp = dz;
		double hp;

		if (dz >= 0.0)
		{
			hp = h1;
			if (l1 < z2)
			{
				dzp = h1;
			}
		}
		else
		{
			hp = h2;
			if (l2 < z1)
			{
				dzp = -h2;
			}
		}


		beta[0] = 0.5 * G * (hp - 0.5 * fabs(dzp)) * dzp / c12;

		//wet-wet correction
		hp = h1 + alpha[0];
		if (eigen[0] * eigen[2] < 0.0 && hp>0.0 && h1 > 0.0 && h2 > 0.0)  //subcritical
		{
			beta[0] = fmax(beta[0], alpha[0] * eigen[0] - h1 * dx * 0.5 / dt);
			beta[0] = fmin(beta[0], -alpha[2] * eigen[2] + h2 * dx * 0.5 / dt);
		}

		beta[1] = 0.0;
		beta[2] = -beta[0];

		for (i = 0; i < 3; i++)
		{
			d1[i] = 0.0;
			d2[i] = 0.0;
		}

		l1 = hp - beta[0] / eigen[0]; //left intermediate state
		l2 = hp + beta[2] / eigen[2]; //right intermediate state

		if (l2 < -EPS12 && h2 < EPS12)   //send mass contributions to the left if right cell is dry
		{
			d1[0] += (eigen[0] * alpha[0] - beta[0]) + (eigen[2] * alpha[2] - beta[2]);
		}
		else
		{
			if (l1 < -EPS12 && h1 < EPS12)   //send mass contributions to the right if left cell is dry
			{
				d2[0] += (eigen[0] * alpha[0] - beta[0]) + (eigen[2] * alpha[2] - beta[2]);
			}
			else
			{
				for (i = 0; i < 3; i++)   //regular contributions
				{
					if (eigen[i] > 0.0)
					{
						d2[0] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][0];
						d2[1] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][1];
						d2[2] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][2];
					}
					else
					{
						d1[0] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][0];
						d1[1] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][1];
						d1[2] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][2];
					}
				}
			}
		}

		for (i = 0; i < 3; i++)   //entropy
		{
			//entropy
			if (eigenE[i] > 0.0)
			{
				d2[0] += (eigenE[i] * alpha[i]) * eigev[i][0];
				d2[1] += (eigenE[i] * alpha[i]) * eigev[i][1];
				d2[2] += (eigenE[i] * alpha[i]) * eigev[i][2];
			}
			else
			{
				d1[0] += (eigenE[i] * alpha[i]) * eigev[i][0];
				d1[1] += (eigenE[i] * alpha[i]) * eigev[i][1];
				d1[2] += (eigenE[i] * alpha[i]) * eigev[i][2];
			}
		}

		for (i = 0; i < 3; i++)
		{
			if (fabs(d1[i]) < EPS12)
			{
				d1[i] = 0.0;
			}
			if (fabs(d2[i]) < EPS12)
			{
				d2[i] = 0.0;
			}
		}

		rhsh0->Set(id1, jj, dt * d1[0] / dx);
		rhsqx0->Set(id1, jj, dt * d1[1] / dx);
		rhsqy0->Set(id1, jj, dt * d1[2] / dx);

		rhsh1->Set(id2, jj, dt * d2[0] / dx);
		rhsqx1->Set(id2, jj, dt * d2[1] / dx);
		rhsqy1->Set(id2, jj, dt * d2[2] / dx);
	}
}


PARALLEL void APPFIS::STENCIL(const int ii, const int jj, double dx, double dt, double hextra, int dummy,
	Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* sqrth,
	Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1)
{
	int id1, id2;
	double nx, ny;

	id1 = jj;
	id2 = jj - 1;
	nx = 0.0;
	ny = 1.0;

	double h1 = h_arr->Get(ii, id1);
	double h2 = h_arr->Get(ii, id2);

	if (h1 > 0.0 || h2 > 0.0)
	{
		int i;
		double alpha[3], beta[3], eigen[3], eigenE[3], eigev[3][3], d1[3], d2[3];
		double u1, u2, v1, v2;

		double qx1 = qx_arr->Get(ii, id1);
		double qy1 = qy_arr->Get(ii, id1);
		double z1 = dem->Get(ii, id1);

		double qx2 = qx_arr->Get(ii, id2);
		double qy2 = qy_arr->Get(ii, id2);
		double z2 = dem->Get(ii, id2);

		if (h1 >= hextra)
		{
			u1 = qx1 / h1;
			v1 = qy1 / h1;
		}
		else
		{
			u1 = 0.0;
			v1 = 0.0;
		}
		if (h2 >= hextra)
		{
			u2 = qx2 / h2;
			v2 = qy2 / h2;
		}
		else
		{
			u2 = 0.0;
			v2 = 0.0;
		}

		double sqh1 = sqrth->Get(ii, id1);
		double sqh2 = sqrth->Get(ii, id2);

		double h12 = 0.5 * (h1 + h2);
		double dh = h2 - h1;
		double c12 = sqrt(G * h12);
		double u12 = (u1 * sqh1 + u2 * sqh2) / (sqh1 + sqh2);
		double v12 = (v1 * sqh1 + v2 * sqh2) / (sqh1 + sqh2);
		double un = u12 * nx + v12 * ny;

		eigen[0] = un - c12;
		eigen[1] = un;
		eigen[2] = un + c12;

		//entropy correction
		for (i = 0; i < 3; i++)
		{
			eigenE[i] = 0.0;
		}

		double e1 = u1 * nx + v1 * ny - SQRTG * sqh1;
		double e2 = u2 * nx + v2 * ny - SQRTG * sqh2;
		if (e1 < 0.0 && e2>0.0)
		{
			eigenE[0] = eigen[0] - e1 * (e2 - eigen[0]) / (e2 - e1);
			eigen[0] = e1 * (e2 - eigen[0]) / (e2 - e1);
		}

		e1 = u1 * nx + v1 * ny + SQRTG * sqh1;
		e2 = u2 * nx + v2 * ny + SQRTG * sqh2;

		if (e1 < 0.0 && e2>0.0)
		{
			eigenE[2] = eigen[2] - e2 * (eigen[2] - e1) / (e2 - e1);
			eigen[2] = e2 * (eigen[2] - e1) / (e2 - e1);
		}

		eigev[0][0] = 1.0;
		eigev[0][1] = u12 - c12 * nx;
		eigev[0][2] = v12 - c12 * ny;
		eigev[1][0] = 0.0;
		eigev[1][1] = -c12 * ny;
		eigev[1][2] = c12 * nx;
		eigev[2][0] = 1.0;
		eigev[2][1] = u12 + c12 * nx;
		eigev[2][2] = v12 + c12 * ny;

		alpha[0] = 0.5 * (dh - (((qx2 - qx1) * nx + (qy2 - qy1) * ny) - un * dh) / c12);
		alpha[1] = (((qy2 - qy1) - v12 * dh) * nx - ((qx2 - qx1) - u12 * dh) * ny) / c12;
		alpha[2] = 0.5 * (dh + (((qx2 - qx1) * nx + (qy2 - qy1) * ny) - un * dh) / c12);

		double dz = z2 - z1;
		double l1 = z1 + h1;
		double l2 = z2 + h2;
		double dzp = dz;
		double hp;

		if (dz >= 0.0)
		{
			hp = h1;
			if (l1 < z2)
			{
				dzp = h1;
			}
		}
		else
		{
			hp = h2;
			if (l2 < z1)
			{
				dzp = -h2;
			}
		}

		beta[0] = 0.5 * G * (hp - 0.5 * fabs(dzp)) * dzp / c12;


		//wet-wet correction
		hp = h1 + alpha[0];
		if (eigen[0] * eigen[2] < 0.0 && hp>0.0 && h1 > 0.0 && h2 > 0.0)  //subcritical
		{
			beta[0] = fmax(beta[0], alpha[0] * eigen[0] - h1 * dx * 0.5 / dt);
			beta[0] = fmin(beta[0], -alpha[2] * eigen[2] + h2 * dx * 0.5 / dt);
		}

		beta[1] = 0.0;
		beta[2] = -beta[0];

		for (i = 0; i < 3; i++)
		{
			d1[i] = 0.0;
			d2[i] = 0.0;
		}

		l1 = hp - beta[0] / eigen[0]; //left intermediate state
		l2 = hp + beta[2] / eigen[2]; //right intermediate state

		if (l2 < -EPS12 && h2 < EPS12)   //send mass contributions to the left if right cell is dry
		{
			d1[0] += (eigen[0] * alpha[0] - beta[0]) + (eigen[2] * alpha[2] - beta[2]);
		}
		else
		{
			if (l1 < -EPS12 && h1 < EPS12)   //send mass contributions to the right if left cell is dry
			{
				d2[0] += (eigen[0] * alpha[0] - beta[0]) + (eigen[2] * alpha[2] - beta[2]);
			}
			else
			{
				for (i = 0; i < 3; i++)   //regular contributions
				{
					if (eigen[i] > 0.0)
					{
						d2[0] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][0];
						d2[1] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][1];
						d2[2] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][2];
					}
					else
					{
						d1[0] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][0];
						d1[1] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][1];
						d1[2] += (eigen[i] * alpha[i] - beta[i]) * eigev[i][2];
					}
				}
			}

		}

		for (i = 0; i < 3; i++)   //entropy
		{
			//entropy
			if (eigenE[i] > 0.0)
			{
				d2[0] += (eigenE[i] * alpha[i]) * eigev[i][0];
				d2[1] += (eigenE[i] * alpha[i]) * eigev[i][1];
				d2[2] += (eigenE[i] * alpha[i]) * eigev[i][2];
			}
			else
			{
				d1[0] += (eigenE[i] * alpha[i]) * eigev[i][0];
				d1[1] += (eigenE[i] * alpha[i]) * eigev[i][1];
				d1[2] += (eigenE[i] * alpha[i]) * eigev[i][2];
			}
		}

		for (i = 0; i < 3; i++)
		{
			if (fabs(d1[i]) < EPS12)
			{
				d1[i] = 0.0;
			}
			if (fabs(d2[i]) < EPS12)
			{
				d2[i] = 0.0;
			}
		}

		//in the Y-solver flux we have to accumulate over rhs to NOT overwrite the X-flux values
		rhsh0->Set(ii, id1, rhsh0->Get(ii, id1) + dt * d1[0] / dx);
		rhsqx0->Set(ii, id1, rhsqx0->Get(ii, id1) + dt * d1[1] / dx);
		rhsqy0->Set(ii, id1, rhsqy0->Get(ii, id1) + dt * d1[2] / dx);

		rhsh1->Set(ii, id2, rhsh1->Get(ii, id2) + dt * d2[0] / dx);
		rhsqx1->Set(ii, id2, rhsqx1->Get(ii, id2) + dt * d2[1] / dx);
		rhsqy1->Set(ii, id2, rhsqy1->Get(ii, id2) + dt * d2[2] / dx);
	}
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, double dt, double hextra,
	Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem, Array<double>* n_arr,
	Array<double>* rhsh0, Array<double>* rhsh1, Array<double>* rhsqx0, Array<double>* rhsqx1, Array<double>* rhsqy0, Array<double>* rhsqy1)
{
	double hij, qxij, qyij;

	double hn = h_arr->Get(i, j);

	//update depth
	hij = hn - rhsh0->Get(i, j) - rhsh1->Get(i, j);

	//if there are negative depths, there are removed (for the moment). Should be residual values (close to machine accuracy but negative)
	if (hij < 0.0)
	{
		hij = 0.0;
	}

	//if water is below hextra, velocities are removed
	if (hij < hextra)
	{
		qxij = 0.0;
		qyij = 0.0;
	}
	else
	{
		//update friction and momentum
		double mx = qx_arr->Get(i, j) - rhsqx0->Get(i, j) - rhsqx1->Get(i, j);
		double my = qy_arr->Get(i, j) - rhsqy0->Get(i, j) - rhsqy1->Get(i, j);

		double modM = sqrt(mx * mx + my * my);
		if (n_arr->Get(i, j) > EPS12 && hn >= hextra && modM > EPS12)
		{
			double tt = dt * G * n_arr->Get(i, j) * modM / (hn * hn * cbrt(hn));
			qxij = -0.5 * (mx - mx * sqrt(1.0 + 4.0 * tt)) / tt;
			qyij = -0.5 * (my - my * sqrt(1.0 + 4.0 * tt)) / tt;
		}
		else
		{
			qxij = mx;
			qyij = my;
		}
	}

	h_arr->Set(i, j, hij);
	qx_arr->Set(i, j, qxij);
	qy_arr->Set(i, j, qyij);
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, double dt, double hextra,
	Array<double>* h_arr, Array<double>* qx_arr, Array<double>* qy_arr, Array<double>* dem)
{
	double hij = h_arr->Get(i, j);
	double zij = dem->Get(i, j);

	if (hij > hextra)
	{
		if (((hij + zij < dem->Get(i + 1, j)) && (h_arr->Get(i + 1, j) < EPS12)) || ((hij + zij < dem->Get(i - 1, j)) && (h_arr->Get(i - 1, j) < EPS12)))
		{
			qx_arr->Set(i, j, 0.0);
		}
		if (((hij + zij < dem->Get(i, j - 1)) && (h_arr->Get(i, j - 1) < EPS12)) || ((hij + zij < dem->Get(i, j + 1)) && (h_arr->Get(i, j + 1) < EPS12)))
		{
			qy_arr->Set(i, j, 0.0);
		}
	}
}


PARALLEL void APPFIS::STENCIL(const int i, int size, double cellsize, double global_dt, double simtime, int idx_low, int idx_high,
	Array<int>* src_g, Array<double>* src_time, Array<double>* src_val, Array<double>* h_arr)
{
	double flow_at_idx_low = src_val->Get(idx_low * size + i);
	double flow_at_idx_high = src_val->Get(idx_high * size + i);
	double time_at_idx_low = src_time->Get(idx_low);
	double time_at_idx_high = src_time->Get(idx_high);

	double flow;

	if (idx_low == idx_high)
	{
		flow = flow_at_idx_low;
	}
	else
	{
		double time_diff = simtime - time_at_idx_low;
		double time_diff_2 = time_at_idx_high - time_at_idx_low;
		flow = flow_at_idx_low + (((flow_at_idx_high - flow_at_idx_low) * time_diff) / time_diff_2);
	}

	int sid = src_g->Get(i);
	double hij = h_arr->Get(sid);
	double h_src = (flow * global_dt) / (cellsize * cellsize);
	hij += h_src;
	h_arr->Set(sid, hij);
}

#endif