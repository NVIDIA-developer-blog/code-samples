#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// 
// Sample demonstrating OpenGL context creation without X server 
// 
// This associates the parallel 4 all blog post 
// "EGL Vision: OpenGL Visualization without X server"

// Some choices to select the EGL display and the render target
//
// use eglGetDisplay or device/platform extensions to select display
#define USE_EGL_GET_DISPLAY
//#undef USE_EGL_GET_DISPLAY

// use egl surface or manually managed FBO as render target
#define USE_EGL_SURFACE
//#undef USE_EGL_SURFACE


#ifdef USE_EGL_GET_DISPLAY
#include <EGL/egl.h>
#else
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <GL/gl.h>

  static const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_DEPTH_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
  };


  static const int pbufferWidth = 9;
  static const int pbufferHeight = 9;

  static const EGLint pbufferAttribs[] = {
        EGL_WIDTH, pbufferWidth,
        EGL_HEIGHT, pbufferHeight,
        EGL_NONE,
  };


// use eglGetDisplay or device/platform extensions to select display
#define USE_EGL_GET_DISPLAY
//#undef USE_EGL_GET_DISPLAY

// use egl surface or manually managed FBO as render target
#define USE_EGL_SURFACE
//#undef USE_EGL_SURFACE

int main(int argc, char *argv[])
{

    // 1. Initialize EGL
#ifdef USE_EGL_GET_DISPLAY
    // obtain the display via default display. 
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
#else
    // obtain display by specifying an appropriate device. This is the preferred method today.

    // load the function pointers for the device,platform extensions 
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
               (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT) { 
         printf("ERROR: extension eglQueryDevicesEXT not available"); 
         return(-1); 
    } 
    
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
               (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if(!eglGetPlatformDisplayEXT) { 
         printf("ERROR: extension eglGetPlatformDisplayEXT not available"); 
         return(-1);  
    }
  
    static const int MAX_DEVICES = 16;
    EGLDeviceEXT devices[MAX_DEVICES];
    EGLint numDevices;

    eglQueryDevicesEXT(MAX_DEVICES, devices, &numDevices);

    EGLDisplay eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], 0);
#endif    

    EGLint major, minor;

    eglInitialize(eglDpy, &major, &minor);

    // 2. Select appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 3. Bind the API 
    eglBindAPI(EGL_OPENGL_API);

    // 4. create the context
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);

#ifdef USE_EGL_SURFACE
    // 5. create the surface and make the context current current
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
#else
    // 5. make the context current without a surface
    eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);

    // manually create the OpenGL buffers and textures: A  framebuffer object with associated 
    // texture and depth buffers. 

    // create a 2D texture as the color buffer
    GLuint texId;
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D,  texId);
    glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGBA8,  pbufferWidth, pbufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);    

    // create a renderbuffer for storing the depth component
    GLuint rbId;
    glGenRenderbuffers(1, &rbId);
    glBindRenderbuffer(GL_RENDERBUFFER, rbId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, pbufferWidth, pbufferHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // create the framebuffer and attach the texture (color buffer) and the renderbuffer (depth buffer)
    GLuint fbId;
    glGenFramebuffers(1, &fbId);
    glBindFramebuffer(GL_FRAMEBUFFER, fbId);    

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbId);

    // set the buffer dimensions
    glViewport(0, 0, pbufferWidth, pbufferHeight);
#endif
    
    // Rendering a basic triangle
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    glColor3f(0.1, 0.2, 0.3);

    glVertex3f(-0.5, 0.5, 0);
    glVertex3f( 0.5, 0.5, 0);
    glVertex3f( 0, -0.5, 0);
    glEnd();

    // Read the triangle data back into a buffer
    unsigned char* data = (unsigned char*) malloc(4 * pbufferWidth * pbufferHeight);

    glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, data);


    // write  the data out in ppm format
    printf("P3\n");
    printf("%d %d\n", pbufferWidth, pbufferHeight);
    printf("255 \n");
    int i=0;
    for(int y=0; y<pbufferHeight; y++){
      for(int x=0; x<pbufferWidth; x++) {
          printf("%2d %2d %2d ",(int) data[i+2],(int) data[i+1], (int) data[i+0]);
          i+=4;
      }
     printf("\n");
    }

    // cleanup
    free(data);

    eglTerminate(eglDpy);
    return 0;
}

