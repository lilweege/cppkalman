example: example.cpp cppkalman.hpp
	g++ -std=c++20 -Wall -Wextra -Wshadow -Wconversion -Wpedantic -Ofast -DEIGEN_NO_DEBUG -isystem eigen example.cpp