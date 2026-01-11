#include <cuda_runtime_api.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <deque>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace std;

std::deque<char*> files;
std::string args;

void parseOptions(int argc, char** argv)
{
	bool options = false;

	while (argc-- > 1)
	{
		if (strcmp(argv[argc], "--") == 0)
		{
			options = true;
		}
		else if (options)
		{
			args.insert(0, " ").insert(0, argv[argc]);
		}
		else
		{
			files.push_front(argv[argc]);
		}
	}
}

int verifyCudaAvailability()
{
	int count;
	switch (cudaGetDeviceCount(&count))
	{
		case cudaErrorNoDevice:
			cerr << "No CUDA capable device found!" << endl;
			exit(1);
		case cudaErrorInsufficientDriver:
			cerr << "CUDA drivers not available!" << endl;
			exit(1);
		default: break;
	}

	return count;
}

std::string filename(std::string const& s)
{
    std::stringstream ss(s);
    std::string file;
    while (std::getline(ss, file, '/'));

    return file;
}

void insertNewline()
{
	cout << endl;
}

int main(int argc,char **argv)
{
	int cudaDevices = verifyCudaAvailability();

	atexit(insertNewline);

	parseOptions(argc, argv);

	std::ostringstream command;

#if MPI_SUPPORT
	if (cudaDevices > 1)
	{
		command << "mpirun -np " << cudaDevices << " ";
	}
#endif

	command << "./hairrazor " << args;

	for (int i = 0; i < files.size(); ++i)
	{
		int percentage = float(i)/files.size()*100;

		cout << "\33[2K\r";
		cout << std::setw(5) << percentage << '%' << '\t';
		cout << std::setw(5) << (i + 1) << '/' << files.size() << '\t';
		cout << files[i] << std::flush;

		int status = system(
			command.str()
				.append(" -f ").append(files[i])
				.append(" -o output/").append(filename(files[i]))
				.c_str()
		);

		if (status != 0) return status;
	}

	cout << "\33[2K\r";
	cout << std::setw(5) << 100 << '%' << '\t';
	cout << std::setw(5) << files.size() << '/' << files.size() << std::flush;

	return 0;
}
