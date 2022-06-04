/** @file Utils.hpp
*  @brief Header containing the custom utilities
*
*  This contains the subroutines and eventually any
*  methods to help custom operation
*
*  @author Md Bulbul Sharif
*/


#ifndef UTILS_HPP
#define UTILS_HPP


#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <sstream>
#include <sys/stat.h>
#include <iostream>

#ifdef _WIN32
#include "Dirent.hpp"
#include <direct.h>
#include <functional>
#define STRNCASECMP _strnicmp
#else
#include <dirent.h>
#define STRNCASECMP strncasecmp
#endif


namespace APPFIS
{

	typedef std::vector<std::string> StringVector;    /**< Custom vector type that represents a vector of strings. */


	/** @brief It splits a string by a char delimeter.
	*
	*  @param s String
	*  @param delim Char delimeter
	*  @param elems String vector
	*  @return Splited string vector
	*/
	StringVector& Split(const std::string& s, char delim, StringVector& elems);


	/** @brief It splits a string by a char delimeter.
	*
	*  @param s String
	*  @param delim Char delimeter
	*  @return Splited string vector
	*/
	StringVector Split(const std::string& s, char delim);


	/** @brief It converts every element of a string vector into a specific type point vector.
	*
	*  @param vs String vector
	*  @return Specific type point vector
	*/
	template <typename T>
	std::vector<T> VecstrToVectype(StringVector vs);


	/** @brief It trims left side of a string.
	*
	*  @param s Input string
	*  @return Left trimmed string
	*/
	static inline std::string& LeftTrim(std::string& s)
	{
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
		return s;
	}


	/** @brief It trims right side of a string.
	*
	*  @param s Input string
	*  @return Right trimmed string
	*/
	static inline std::string& RightTrim(std::string& s)
	{
		s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
		return s;
	}


	/** @brief It trims a string from both side.
	*
	*  @param s Input string
	*  @return Trimmed string
	*/
	static inline std::string& Trim(std::string& s)
	{
		return LeftTrim(RightTrim(s));
	}


	/** @brief It computes the parent directory from the path.
	*
	*  @param path The full path of a file
	*  @return A string contains the parent directory
	*/
	std::string GetParentDir(const char* path);


	/** @brief It creates a directory if not available.
	*
	*  @param path The full path of a directory
	*/
	void CreateDir(std::string path);


	int StringRemoveDelimiter(char delimiter, const char* string);
	bool CheckArgument(int argc, char* argv[], const char* string_ref);
	int GetArgument(int argc, char* argv[], const char* string_ref);


	StringVector& Split(const std::string& s, char delim, StringVector& elems)
	{
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim))
		{
			elems.push_back(item);
		}
		return elems;
	}


	StringVector Split(const std::string& s, char delim)
	{
		std::vector<std::string> elems;
		return Split(s, delim, elems);
	}


	template <typename T>
	std::vector<T> VecstrToVectype(StringVector vs)
	{
		std::vector<T> ret;
		for (StringVector::iterator it = vs.begin(); it != vs.end(); ++it)
		{
			std::istringstream iss(*it);
			T temp;
			iss >> temp;
			ret.push_back(temp);
		}
		return ret;
	}


	std::string GetParentDir(const char* path)
	{
		std::string spath(path);
		size_t found = spath.find_last_of("/\\");

		if (found == spath.length() - 1)
		{
			spath = spath.substr(0, found);
		}

		return spath.substr(0, (spath.find_last_of("/\\")));
	}


	void CreateDir(std::string path)
	{
		if (path.empty()) return;

		size_t dot = path.find_last_of(".");
		size_t slash = path.find_last_of("/\\");

		if (dot != std::string::npos)
		{
			if (slash != std::string::npos)
			{
				if (dot > slash) return;
			}
			else
			{
				return;
			}
		}

		DIR* dir = opendir(path.c_str());
		if (!dir)
		{
#ifdef _WIN32
			if (_mkdir(path.c_str()) != 0)
			{
				std::cerr << "There is an error" << std::endl;
			}
#else
			mkdir(path.c_str(), S_IRWXU);
#endif
		}
		else
		{
			closedir(dir);
		}
	}


	int StringRemoveDelimiter(char delimiter, const char* string) 
	{
		int string_start = 0;

		while (string[string_start] == delimiter) {
			string_start++;
		}

		if (string_start >= static_cast<int>(strlen(string) - 1)) {
			return 0;
		}

		return string_start;
	}


	bool CheckArgument(int argc, char* argv[], const char* string_ref)
	{
		bool bFound = false;

		if (argc >= 1) {
			for (int i = 1; i < argc; i++) {
				int string_start = StringRemoveDelimiter('-', argv[i]);
				const char* string_argv = &argv[i][string_start];

				const char* equal_pos = strchr(string_argv, '=');
				int argv_length = static_cast<int>(
					equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

				int length = static_cast<int>(strlen(string_ref));

				if (length == argv_length &&
					!STRNCASECMP(string_argv, string_ref, length)) {
					bFound = true;
					continue;
				}
			}
		}

		return bFound;
	}


	int GetArgument(int argc, char* argv[], const char* string_ref)
	{
		bool bFound = false;
		int value = -1;

		if (argc >= 1) {
			for (int i = 1; i < argc; i++) {
				int string_start = StringRemoveDelimiter('-', argv[i]);
				const char* string_argv = &argv[i][string_start];
				int length = static_cast<int>(strlen(string_ref));

				if (!STRNCASECMP(string_argv, string_ref, length)) {
					if (length + 1 <= static_cast<int>(strlen(string_argv))) {
						int auto_inc = (string_argv[length] == '=') ? 1 : 0;
						value = atoi(&string_argv[length + auto_inc]);
					}
					else {
						value = 0;
					}

					bFound = true;
					continue;
				}
			}
		}

		if (bFound) {
			return value;
		}
		else {
			return 0;
		}
	}

}


#endif