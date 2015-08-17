function ArrayfunArticle
%% Making the most of the GPU from MATLAB
%
% In this article we will discuss the techniques you should be using to
% maximize the performance of your GPU-accelerated MATLAB(R) code.
%
% First we explain how to write MATLAB code which is inherently
% parallelizable. This technique, known as _vectorization_, benefits all
% your code whether or not it uses the GPU.
%
% We then present a family of function wrappers - |bsxfun|, |pagefun| and
% |arrayfun| - that take advantage of GPU hardware, yet require no
% specialist parallel programming skills. The most advanced function,
% |arrayfun|, allows you to write your own custom kernels in the MATLAB
% language.
%
% If these techniques do not provide the performance or flexibility you
% were after, you can still write custom CUDA code in C or C++ that you can
% run from MATLAB, as discussed in our earlier articles on
% <http://devblogs.nvidia.com/parallelforall/prototyping-algorithms-and-testing-cuda-kernels-matlab/
% |CUDAKernels|> and
% <http://devblogs.nvidia.com/parallelforall/calling-cuda-accelerated-libraries-matlab-computer-vision-example/
% MEX functions>.
%
% All of the features described here are available out of the box with
% MATLAB and Parallel Computing Toolbox(TM).


%% Mobile phone signal strength example
% We will use a single example that will help us to illustrate these
% techniques.
%
% A cellular phone network wants to map its coverage to help plan for new
% antenna installations. We imagine an idealized scenario with M = 25
% cellphone masts, each H = 20 meters in height, evenly spaced on an
% undulating terrain.
%
% On the GPU, we define a number of variables including:
%
% * |map|: An N x 3 height field in a 10km x 10km grid (N = 10,000)
% * |masts|: An M x 3 array of antenna positions, at height |H|
% * |AntennaDirection|: A 3 x M array of vectors representing the
% orientation of each antenna.

% Map definition
gridpts = linspace(-5, 5, 100);
[mapX, mapY] = meshgrid(gridpts*1000);
N = numel(mapX);
% Procedurally generated terrain
mapZ = 100 * (peaks(gridpts/2) + 0.3*peaks(gridpts/2)' + flipud(peaks(gridpts/6)));

% Antenna masts - index into map spacing every 20 gridpoints
index = 1:20:100;
mastX = mapX(index, index);
mastY = mapY(index, index);
H = 20; % All masts are 20 meters in height
mastZ = mapZ(index, index) + H;
% Antenna properties
M = numel(mastX);
Frequency = 800e6;         % 800 MHz transmitters
Power = 100 * ones(1, M);  % Most transmitters use 100 W of power
Power(1:4:M) = 400;        % A few transmitters are more powerful
Power([3 14 22]) = 0;      % A small number are out of order
% Finally, give each antenna a random orientation. This is represented by
% horizontal vectors representing the direction the antennae are facing.
AntennaAngle = rand(1, M) * 2*pi;
AntennaDirection = [cos(AntennaAngle); sin(AntennaAngle); zeros(1, M)];

% Set up some random rotation matrices, stacked along the 3rd dimension as
% 3 x 3 x M arrays
tiltAngle = gpuArray.rand([1 1 M])*360;
Zero = gpuArray.zeros([1 1 M]);
One  = gpuArray.ones([1 1 M]);
Tilt = [One  Zero             Zero;
        Zero cosd(tiltAngle) -sind(tiltAngle);
        Zero sind(tiltAngle)  cosd(tiltAngle)];
turnAngle = gpuArray.rand([1 1 M])*360;
Pan = [cosd(turnAngle)  -sind(turnAngle) Zero;
       sind(turnAngle)   cosd(turnAngle) Zero;
       Zero              Zero            One];

% Set up indices into the data
mapIndex = gpuArray.colon(1,N)'; % N x 1 array of map indices
mastIndex = gpuArray.colon(1,M); % 1 x M array of mast indices
[RowIndex, ColIndex] = ndgrid(mapIndex, mastIndex);

% Put the map data on the GPU and concatenate the map positions into a
% single 3-column matrix containing all the coordinates [X, Y, Z].
map = gpuArray([mapX(:) mapY(:) mapZ(:)]);
masts = gpuArray([mastX(:) mastY(:) mastZ(:)]);
AntennaDirection = gpuArray(AntennaDirection);

%%
% This is what the map looks like:
drawMap(map, masts, AntennaDirection, H);

%% Vectorization basics
% Inefficiencies in code generally appear as loops or repeated segments of
% code, where the recurring operations are naturally parallel - they don't
% depend on each other.
%
% The simplest kind of vectorization is to take advantage of matrix algebra
% in your mathematical operations. Let's say I'm trying to rotate all the
% antennae downwards by 10 degrees:

% 3D rotation around the X axis
Elevation = [1        0         0;
             0 cosd(10) -sind(10);
             0 sind(10)  cosd(10)];
% Allocate a new array of directions and loop through to compute them
NewAntennaDirection = gpuArray.zeros(size(AntennaDirection));
for ii = 1:M
    NewAntennaDirection(:,ii) = Elevation * AntennaDirection(:,ii);
end

%%
% Note, however, that there's no dependency between one rotation and the
% next. The rules of matrix multiplication let me do this without a loop:

AntennaDirection = Elevation * AntennaDirection;

%%
% This will then run a lot faster, especially on the GPU which was crippled
% by very low utilization in the serial code (we were asking it to do too
% little work at a time).
%
% The following code calculates the peak signal strength at each gridpoint
% on the map by measuring the losses between the gridpoint and the
% transmitters. It loops over every gridpoint and every antenna, computing
% one value per gridpoint. Since modern cellphones are in communication
% with multiple transmitters at once, we take an average of the three
% strongest signals. To add some additional real-world complexity to the
% calculation, antennae that are pointing away from the location cannot be
% included (the signal strength is effectively zero).

% Allocate GPU memory for results
signalMap = gpuArray.zeros(size(mapX));
signalPowerDecibels = gpuArray.zeros(M, 1);
tic; % Start timer

% Outer loop over the gridpoints
NN = 10; % This version is too slow to process more than a few points
for ii = 1:NN

    % Get the power received from every mast at this gridpoint
    for jj = 1:M

        % Calculate the distance from map position ii to antenna jj
        pathToAntenna = masts(jj,:) - map(ii,:);
        distance = sqrt(sum(pathToAntenna).^2);
        % Apply the free space path loss formula to the antenna power
        pathLoss = (4 .* pi .* distance * Frequency ./ 3e8).^2;
        signalPowerWatts = Power(jj) ./ pathLoss;
        % Convert to decibels (dBm)
        signalPowerDecibels(jj) = 30 + 10 * log10(signalPowerWatts);
        
        % Reset the power to zero if the antenna isn't facing this way.
        % We can tell this from the dot product.
        directionDotProduct = dot(pathToAntenna, AntennaDirection(:,jj));
        if directionDotProduct < 0
            signalPowerDecibels(jj) = -inf; % 0 Watts = -inf decibels
        end
        
    end
    
    % Sort the power from each mast
    signalPowerSorted = sort(signalPowerDecibels, 'descend');
    % Strength is the average of the three strongest signals
    signalMap(ii) = mean(signalPowerSorted(1:3));
end
loopT = reportTime(toc, 'Signal strength compute time per gridpoint using loops', 1/NN);

%%
% First let's focus on pulling the basic algebra out of the loop. We
% recognize that our scalar operators like |.*| and |sqrt| will work on
% larger arrays in an element-by-element manner. We also recognize that
% _reductions_, like |sum| and |mean|, can work along a chosen dimension of
% an N-dimensional input. Using these features and reshaping the data
% allows us to remove the loops.
%
% Let's start by reshaping the data. At the moment we have two lists of
% points, each row representing one point, and the first column
% representing the $x$ coordinate, the second $y$, and the third $z$. Let's
% shift the three Cartesian coordinates to the 3rd dimension, and instead
% use rows and columns to differentiate between gridpoints and antenna
% masts.

tic; % Start timer
X = reshape(map, [], 1, 3);
A = reshape(masts, 1, [], 3);

%%
% Let us say $X_i = [X^x_i, X^y_i, X^z_i]$ represents the $i$'th map
% position, and $A_j = [A^x_j, A^y_j, A^z_j]$ the $j$'th antenna position,
% noting that these are 1 x 1 x 3 arrays with the elements packed along the
% third dimension. Now the map data is a single column with one gridpoint
% per row, and the antenna data has a single row, with one antenna per
% column:
%
% $$ X = \left[ \begin{array}{c}
%     X_1 \\
%     X_2 \\
%     \vdots \\
%     X_N \end{array} \right] ,
% ~~
%    A = \left[ \begin{array}{cccc} A_1 & A_2 & \cdots & A_M \end{array} \right]$$
%
% We want to create a matrix of size N x M which contains the distance
% of every gridpoint to every antenna. Conceptually, we want to replicate
% the map data along the columns, and the antenna data along the rows, and
% subtract the two arrays to get the path vectors from gridpoints to
% antennae:
%
% $$\left[ \begin{array}{llll}
%          A_1 & A_2 & \cdots & A_M \\[0.7em] \vdots & \vdots & \vdots &
%          \vdots \\[0.7em] A_1 & A_2 & \cdots & A_M \end{array}
%   \right] - \left[ \begin{array}{lll}
%          X_1 & \cdots & X_1 \\ X_2 & \cdots & X_2 \\ \vdots & \cdots &
%          \vdots \\ X_N & \cdots & X_N \end{array}
%   \right]$$
%
% We could do this using |repmat|:
%
%    pathToAntenna = repmat(A, [N, 1, 1]) - repmat(X, [1, M, 1]);
%
% However, this is actually doing unnecessary work and taking up extra
% memory. We introduce the function
% <http://www.mathworks.com/help/distcomp/bsxfun.html |bsxfun|>, the first
% in a family of essential function wrappers useful for high performance
% coding. |bsxfun| applies a binary (two-input) operation (such as |minus|
% in this case), expanding along any dimensions that don't match between
% the two arguments as it goes, extending the normal rules for scalars.

pathToAntenna = bsxfun(@minus, A, X);

%%
% Computing the path loss involves scalar operations that work
% independently on each element of an array (known as _element-wise_), plus
% <http://www.mathworks.com/help/matlab/ref/sum.html |sum|> to add the x, y
% and z values along the 3rd dimension.
distanceSquared = sum(pathToAntenna.^2, 3); % Syntax to sum along dim 3
distance = sqrt(distanceSquared);
pathLoss = (4 .* pi .* distance .* Frequency ./ 3e8).^2;

%%
% The power and dot product calculations can also be done using
% combinations of |bsxfun|, |sum|, and element-wise arithmetic.

% Power calculation
signalPowerWatts = bsxfun(@rdivide, Power, pathLoss);
signalPowerDecibels = 30 + 10 * log10(signalPowerWatts);

% Dot product is just the sum along X,Y,Z of two arrays multiplied
dirn = permute(AntennaDirection, [3 2 1]); % Permute to 1 x M x 3
directionDotProduct = sum(bsxfun(@times, pathToAntenna, dirn), 3);

%%
% The looping code included a conditional statement to set the power to
% zero (or $-\infty$ decibels) if the antenna was facing the wrong way. The
% array solution is to compute a logical index:
isFacing = directionDotProduct >= 0;

%%
% This is a 'mask' identifying every entry in the data to be included in
% the calculation. What is more it can be used to index all the invalid
% entries and set them to $-\infty$ so they will be sorted to the end.
signalPowerDecibels(~isFacing) = -inf;

%%
% Finally, to compute the signal strength we have to tell
% <http://www.mathworks.com/help/matlab/ref/sort.html |sort|> and
% <http://www.mathworks.com/help/matlab/ref/mean.html |mean|> to operate
% along each row.
signalPowerSorted = sort(signalPowerDecibels, 2, 'descend');
signalMap = mean(signalPowerSorted(:,1:3), 2);

%%
% Even for this relatively small amount of data (for a GPU application) we
% can see a significant speed-up. This is partly because the GPU was being
% used so inefficiently before - with thousands of kernels being launched
% serially in a loop - and partly because the GPU's multiprocessors are
% much more fully utilized.
%
% Here is the resulting timing and a 'heat map' showing the signal strength
% at each point on the map:
reportTime(toc, 'Signal strength compute time per gridpoint after vectorization', 1/N, loopT);
close; drawSignalMap(map, masts, AntennaDirection, H, signalMap)

%% Advanced vectorization
% Vectorizing is advisable for any performance-critical code, whether it
% uses the GPU or not.
% <http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
% MATLAB's documentation> provides a wealth of advice on the different
% techniques to use, the majority of which apply equally well to GPU code.
%
% The trickiest scenario tends to be when the data is divided into groups
% or categories of different sizes. They then cannot easily be given their
% own row, column or page of an array. |gpuArray| supports a number of the
% features that can help with this including linear and logical indexing,
% |find|, |sort| and |accumarray|.
%
% Let us say we refine our example so that each antenna belongs to one of
% three networks and we want a signal map for each one. We use |sort| to
% group all the signal strengths by network, |diff| and |find| to identify
% the boundaries between groups, and |cumsum| to get the average of the
% strongest three signals. You can see the resulting code
% <multipleSignalMaps.m here>.

%% Batching matrix operations using |pagefun|
%
% Many 2-D matrix operations such as multiply and transpose do not
% naturally vectorize, and this can be a bottleneck when you have a large
% number of small matrices to operate on.
% <http://www.mathworks.com/help/distcomp/pagefun.html |pagefun|> provides
% the solution, letting you carry out a large batch of these operations in
% a single call.
%
% Let's say we want to rotate all the antenna masts by a different
% horizontal (pan) and vertical (tilt) rotation. Vectorization cannot solve
% this problem, so we might revert to using a loop to apply the 3 x 3
% rotations that make up each 'slice' of the 3-D |Pan| and |Tilt| arrays:

% Loop over the antennae applying the Pan and Tilt rotations defined
% earlier
tic;
newAntennaDirection = gpuArray.zeros(size(AntennaDirection));
for a = 1:M
    thisMast = AntennaDirection(:,a);
    newAntennaDirection(:,a) = Pan(:,:,a) * Tilt(:,:,a) * thisMast;
end
loopT = reportTime(toc, 'Time to rotate antennae using a loop');

%%
% Translating this into pagefun operations gives a considerable speedup
% even though there are only 25 simultaneous multiplies in this case.

tic;
oldDirection = reshape(AntennaDirection, [3 1 M]); % Make M pages of 3 x 1 vectors
newAntennaDirection = pagefun(@mtimes, Tilt, oldDirection);
newAntennaDirection = pagefun(@mtimes, Pan, newAntennaDirection);
reportTime(toc, 'Time to rotate antennae using pagefun', 1, loopT);

%%
% As well as all the element-wise functions, matrix multiplication and
% transpose, |pagefun| also supports solving small linear systems in batch
% using |mldivide| (the MATLAB backslash |\| operator).

%% Writing kernels using |arrayfun|
% The way that each |gpuArray| function is implemented varies, but
% typically they will launch one or more kernels to do the bulk of the work
% in parallel. Launching kernels is costly, so this is particularly irksome
% when you are doing a series of independent operations on the elements of
% an array, such as arithmetic. On the face of it, a calculation like
%
%   pathLoss = (4 .* pi .* distance .* Frequency ./ 3e8).^2
%
% is going to launch four or five kernels, when really we would prefer just
% to run one kernel that does all the operations.
%
% MATLAB employs various optimizations to minimize this kind of kernel
% launch proliferation. However you may find you get better performance
% when you explicitly identify parts of your code that you know could be
% compiled into a single kernel. You write your kernel in the MATLAB
% language as a function, which you call using the wrapper |arrayfun|. This
% is the last and most advanced of our wrapper family.
%
% By splitting out the X, Y and Z coordinates, a significant portion of the
% power calculation can be separated into a function containing only scalar
% operations:

    function signalPower = powerKernel(mapIndex, mastIndex)
        
        % Implement norm and dot product calculations via loop
        dSq = 0;
        dotProd = 0;
        for coord = 1:3
            path = A(1,mastIndex,coord) - X(mapIndex,1,coord);
            dSq = dSq + path*path;
            dotProd = dotProd + path*dirn(1,mastIndex,coord);
        end
        d = sqrt(dSq);
        
        % Power calculation, with adjustment for being behind antenna
        if dotProd >= 0
            pLoss = (4 .* pi .* d .* Frequency ./ 3e8).^2;
            powerWatts = Power(mastIndex) ./ pLoss;
            signalPower = 30 + 10 * log10(powerWatts);
        else
            signalPower = -inf;
        end
        
    end

%%
% This function represents a kernel run in parallel at every data point, so
% one thread is run for each element of the input arrays.
%
%   signalPower = arrayfun(@powerKernel, mapIndex, mastIndex);
%
% Like |bsxfun|, |arrayfun| takes care of expanding input arrays along
% dimensions that don't match. In this case, our N x 1 map indices and our
% 1 x M antenna indices are expanded to give the N x M output dimensions.
%
% Now that we've taken full control over the way the power calculation is
% parallelized, we can see a further gain over the original code:

% Original power calculation
gpuArrayTime = gputimeit(@() ...
    powerCalculationWithGpuArray(X, A, Power, Frequency, dirn) );

% Power calculation using arrayfun
arrayfunTime = gputimeit(@() ...
    arrayfun(@powerKernel, mapIndex, mastIndex) );

% Resulting performance improvement
disp(['Speedup for power calculation using arrayfun = ' ...
      num2str(gpuArrayTime/arrayfunTime) 'x']);

%%
% It's worth remarking that in order to implement versions of the vector
% norm and dot product inside the kernel it was necessary to use a |for|
% loop, something we were trying to avoid. However, inside the kernel it is
% not significant since we are already running in parallel.
%
% |arrayfun| kernels support scalar functions along with the majority of
% standard MATLAB syntax for looping, branching, and function execution.
% For more detail see
% <http://www.mathworks.com/help/distcomp/run-element-wise-matlab-code-on-a-gpu.html#bsnx7h8-1
% the documentation>.
%
% A really important feature that we've used here is the ability to index
% into arrays defined _outside_ the nested function definition, such as
% |A|, |X| and |Power|. This simplified the kernel because we didn't
% have to pass all the data in, only the indices identifying the grid
% reference and antenna. Indexing these _upvalues_ works inside |arrayfun|
% kernels as long as it returns a single element.
%
% Note that there is a one-time cost to |arrayfun| kernel execution while
% MATLAB compiles the new kernel.

%%
% Why not always write |arrayfun| kernels instead of writing them in
% another language? Well, |arrayfun| launches as many parallel threads as
% it can, but you have no control over launch configuration, or access to
% shared memory. And of course, you cannot call into third party libraries
% as you could from C or C++ code. This is where
% <http://devblogs.nvidia.com/parallelforall/prototyping-algorithms-and-testing-cuda-kernels-matlab/
% |CUDAKernels|> and
% <http://devblogs.nvidia.com/parallelforall/calling-cuda-accelerated-libraries-matlab-computer-vision-example/
% MEX functions> comes into their own.

%% Conclusion
% Vectorization is a key concept for MATLAB programming with big
% performance advantages for both standard and GPU coding. Combining this
% principle with the wealth of built-in functions optimized for the GPU is
% usually enough to take advantage of your device, without needing to learn
% about CUDA or parallel programming concepts. And if that falls short, the
% |arrayfun| function lets you craft custom kernels in the MATLAB language
% to eke a lot more performance from your card.

%% Further reading
%
% * <http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
% MATLAB documentation on vectorization>
% * <http://www.mathworks.com/help/distcomp/graphics-processing-unit-gpu-computing.html
% MATLAB documentation on GPU Computing>
% * <http://devblogs.nvidia.com/parallelforall/prototyping-algorithms-and-testing-cuda-kernels-matlab/
% ParallelForAll Blog article on |CUDAKernels|>
% * <http://devblogs.nvidia.com/parallelforall/calling-cuda-accelerated-libraries-matlab-computer-vision-example/
% ParallelForAll Blog article on GPU MEX functions>
% * <ArrayfunArticle.zip All the code> needed to reproduce the
% examples and plots in this article.

end