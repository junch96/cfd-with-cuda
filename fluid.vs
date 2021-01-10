#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
  
out vec3 ourColor;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor; // vertex data로부터 가져오 컬러 입력을 ourColor에 설정
    if(aPos.r < 0){
        ourColor = vec3(1.0, 0.0, 0.0);
    }
    else{
        ourColor = vec3(0.0, 0.0, 1.0);
    }
    ourColor = aColor;
}