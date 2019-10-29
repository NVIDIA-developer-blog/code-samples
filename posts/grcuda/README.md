# Source for grCUDA NVIDIA Developer Blog

Contains:

- `bindkernel` + launch [example](kernel/saxpy_nvrtc.py)
- [Example 1](R/dbscan.R): RAPIDS cuML DBSCAN from R
- [Example 2](mandelbrot/app.js): ASCII Mandelbrot Set from Express.js

## Installing grCUDA

1. Download GraalVM CE 19.2.1 for Linux `graalvm-ce-linux-amd64-19.2.1.tar.gz`
   from [GitHub](https://github.com/oracle/graal/releases) and untar it in
   your installation directory. Add that the GraalVM `bin` directory to
   he `PATH` environment variable.

   ```bash
   cd <your installation directory>
   tar xfz graalvm-ce-linux-amd64-19.2.1.tar.gz
   export GRAALVM_DIR=`pwd`/graalvm-ce-19.2.1
   export PATH=${GRAALVM_HOME}/bin:${PATH}:${HOME}
   ```

2. Download the grCUDA JAR from [grcuda/releases](https://github.com/NVIDIA/grcuda/releases) and copy it into `languages` directory.

   ```bash
   cd $GRAALVM_DIR/jre/languages
   mkdir grcuda
   cp <download folder>/grcuda-0.1.0.jar grcuda
   ```

3. Test grCUDA in Node.JS from GraalVM.

   ```console
   cd $GRAALVM_DIR/bin
   ./node --jvm --polyglot
   > arr = Polyglot.eval('grcuda', 'int[5]')
   [Array: null prototype] [ 0, 0, 0, 0, 0 ]
   ```

4. Download other GraalVM languages.

   ```bash
   cd $GRAAL_VM/bin
   ./gu available
   ./gu install python R
   ./gu install ruby   # optionally
   ```

## Running bindkernel (NVRTC) Example

**Binding precompiled kernel with bindkernel:**

1. Compile CUDA C kernel using nvcc.

  ```bash
  cd kernel
  nvcc -ptx saxpy.cu    # produces saxpy.ptx
  ```

2. Run Python script.

  ```bash
  graalpython --jvm --polyglot saxpy.py
  ```

**Compiling and binding kernel with buildkernel:**

1. Run Python script.

  ```bash
  graalpython --jvm --polyglot saxpy_nvrtc.py
  ```

## Running RAPIDS R Example

1. Download the RAPIDS cuML package containing the libraries `libcuml.so` and
   `libcuml++.so` from the [rapidsai-nightly conda channel](https://anaconda.org/rapidsai-nightly/libcuml/files). Choose the package that corresponds to the CUDA Toolkit version that is installed on your system.
   For example:

   - CUDA 10.1: [linux-64/libcuml-0.11.0a1191028-cuda10.1_76.tar.bz2](https://anaconda.org/rapidsai-nightly/libcuml/0.11.0a1191028/download/linux-64/libcuml-0.11.0a1191028-cuda10.1_76.tar.bz2)
   - CUDA 10.0: [linux-64/libcuml-0.11.0a1191028-cuda10.0_76.tar.bz2](https://anaconda.org/rapidsai-nightly/libcuml/0.11.0a1191028/download/linux-64/libcuml-0.11.0a1191028-cuda10.0_76.tar.bz2)
   - CUDA 9.2: [linux-64/libcuml-0.11.0a1191028-cuda9.2_76.tar.bz2](https://anaconda.org/rapidsai-nightly/libcuml/0.11.0a1191028/download/linux-64/libcuml-0.11.0a1191028-cuda9.2_76.tar.bz2)

2. Extract the libraries from archive. In the following, let `$CUML_LIB_DIR`
   point to this directory.

  ```bash
   tar xfj libcuml-0.11.0a1191028-cuda10.1_76.tar.bz2 \
     --strip-components 1 -C ${CUML_LIB_DIR} \
     lib/libcuml.so lib/libcuml++.so
  ```

3. Start FastR (R interpreter).

  ```bash
  # make sure CUML_LIB_DIR points to the directory that contains the two libraries
  cd R
  ./run_r.sh
  >
  ```

4. Install the `seriation` package.

  ```R
  install.packages('seriation')
  ```

5. Run `cuml_dbscan.R` script.

  ```R
  source('cuml_dbscan.R')
  ```

## Running the Python Mandelbrot Set Web Application

1. Install required Node.JS packages into `node_modules` directory of the project.

  ```bash
  cd mandelbrot
  npm install
  ```

2. Run Node.JS application.

  ```bash
  node --jvm --polyglot app.js
  ```

3. Point web browser to [localhost:3000](http://localhost:3000).
