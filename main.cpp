#include "solver.h"
#include "glUtil.h"

enum runArchOption
{
    CPU,
    GPU
};

static inline int getIdx(int x, int y, int sizeX, int sizeY)
{
    int re = y * sizeX + x;
    return re;
}

int go(
    runArchOption runOpt,
    int iterMax
) 
{
    /* Play with these constants, if you want */
    const int sizeX = 16 * 14;
    const int sizeY = 16 * 14;
    //const int sizeX = 4;
    //const int sizeY = 4;
    const int tx = 16;
    const int ty = 16;

    const float density = 0.1;
    const float timestep = 0.005;
    
    const int sizeVer = 6 * sizeX * sizeY;
    float* vertices = new float[sizeVer];
    const int sizeIdx = 6 * (sizeX - 1) * (sizeY - 1);
    int* indices = new int[sizeIdx];
    int idx = 0;
    for(int iy=0;iy<sizeY-1;iy++){
        for(int ix=0;ix<sizeX-1;ix++){
            indices[idx++] = getIdx(ix+1,iy+1,sizeX,sizeY); 
            indices[idx++] = getIdx(ix+1,iy,sizeX,sizeY); 
            indices[idx++] = getIdx(ix,iy+1,sizeX,sizeY); 
            indices[idx++] = getIdx(ix+1,iy,sizeX,sizeY); 
            indices[idx++] = getIdx(ix,iy,sizeX,sizeY); 
            indices[idx++] = getIdx(ix,iy+1,sizeX,sizeY);
        }
    }
    assert(idx == sizeIdx);

    solver::FluidSolver *sv = new solver::FluidSolver(sizeX, sizeY, density, tx, ty);
    sv->dump(vertices);

    float time = 0.0;
    
    {//[1]
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }
    GLFWwindow* window = glfwCreateWindow(sizeX, sizeY, "LearnOpenGL", NULL, NULL);
    {//[2]
        if (window == NULL)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, glUtil::framebuffer_size_callback);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }
    }
    GLuint sp = glUtil::LoadShaders("fluid.vs", "fluid.fs");
    unsigned int VBO, VAO, EBO;
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * sizeVer, vertices, GL_DYNAMIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * sizeIdx, indices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    for(int iter = 0;iter<iterMax;iter++){
        glUtil::processInput(window);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < 4; i++) {
            ++iter;
            sv->addInflow(0.45, 0.2, 0.15, 0.03, 1.0, 0.0, 3.0);
            if(runOpt == CPU)
            {
                sv->update_cpu(timestep, false);
            }
            else
            {
                sv->update_gpu(timestep, false);
            }
            
            time += timestep;
            fflush(stdout);
        }
        sv->dump(vertices);
        
        glUseProgram(sp);
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * sizeVer, vertices, GL_DYNAMIC_DRAW);
        
        glDrawElements(GL_TRIANGLES, sizeof(int) * sizeIdx, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glfwTerminate();
    }

    delete [] vertices;

    return 0;
}
int main(){
    double st, ed;
    st = clock();
    {
        int iterMax = 200;
        go(GPU, iterMax);
    }
    ed = clock();            
    printf("Time with gpu %lf(sec)\n",
    (double)(ed - st)/CLOCKS_PER_SEC);

    st = clock();
    {
        int iterMax = 200;
        go(CPU, iterMax);
    }
    ed = clock();            
    printf("Time with cpu %lf(sec)\n",
    (double)(ed - st)/CLOCKS_PER_SEC);
    return 0;
}