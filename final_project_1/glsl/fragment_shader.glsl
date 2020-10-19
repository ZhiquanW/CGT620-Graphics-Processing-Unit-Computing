#version 430 core
out vec4 FragColor;
in vec3 oCol;
in vec3 FragPos;
float ambientStrength = 0.8;
vec3 light_pos=vec3(10.0f,10.0f,20.0f);
vec3 light_color = vec3(1.0f,1.0f,1.0f);
float specularStrength = 0.5;
vec3 viewPos = vec3(0,0,50);
void main() {
    vec3 ambient = ambientStrength * light_color;
    vec3 norm = vec3(0,0,-1.0f);
    vec3 light_dir = normalize(light_pos-FragPos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-light_color, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * light_color;
    vec3 result =(ambient + diffuse + specular) * oCol;
    FragColor =vec4(result,1.0f);
}