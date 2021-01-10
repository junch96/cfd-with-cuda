#ifndef GLUTIL_H
#define GLUTIL_H
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace glUtil{
	void processInput(GLFWwindow *window);

	void framebuffer_size_callback(
		GLFWwindow* window, 
		int width, 
		int height
	);

	int LoadShaders(
		const char * vertex_file_path,
		const char * fragment_file_path
	);
}
#endif