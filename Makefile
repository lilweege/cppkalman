FLAGS=-std=c++17 -fpermissive -Wall -Wextra -Wshadow -Wconversion -Wpedantic -Ofast -flto
INCLUDES=-DEIGEN_NO_DEBUG -isystem eigen

example.out: example.o example_templates.o
	g++ $(FLAGS) $(INCLUDES) $^ -o $@

%.o: %.cpp example_templates.hpp cppkalman.hpp
	g++ $(FLAGS) $(INCLUDES) -c $<
