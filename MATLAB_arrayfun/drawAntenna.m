function drawAntenna(p, h, dir, b, nsections)
% Draws a 3D representation of an antenna

% Draw box on the ground
base = [p+[b;0;0], p+[0;b;0], p+[-b;0;0], p+[0;-b;0], p+[b;0;0]];
drawline(base);

% Draw edges of pyramid
top = p+[0;0;h];
pyramid = [top, base(:,1), top, base(:,2), top, base(:,3), top, base(:,4)];
drawline(pyramid);

% Draw cross on base
crx = [p+[b;0;0], p+[-b;0;0], p+[0;b;0], p+[0;-b;0]];
drawline(crx);

% Draw segments
if nargin < 5
    nsections = floor(h/b);
end
bottom = base;
sh = h/nsections;
for s = 1:nsections-1
    up = p + s*[0;0;sh];
    r = ((h-(s*sh))/h) * b;
    top = [up+[r;0;0], up+[0;r;0], up+[-r;0;0], up+[0;-r;0], up+[r;0;0]];
    
    % Draw top of segment
    drawline(top);
    
    % Draw crosses on each face
    for f = 1:4
        crx = [bottom(:,1), top(:,2), top(:,1), bottom(:,2)];
        drawline(crx);
        top = circshift(top,1,2);
        bottom = circshift(bottom,1,2);
    end
    
    bottom = top;
end

% Draw antenna direction
l = norm(dir);

% Direction perp to dir horizontally
perph = cross(dir, [0;0;1])/2;
% Direction perp to dir vertically
perpv = cross(dir, perph)/l/2;

% Draw 'Image' plane
top = p+[0;0;h];
c = top + dir;
im = [c+perph+perpv, c+perph-perpv, c-perph-perpv, c-perph+perpv];
drawline(im);

% Draw view pyramid
pyramid = [top, im(:,1), top, im(:,2), top, im(:,3), top, im(:,4)];
drawline(pyramid);

end

function drawline(coords)
line(coords(1,:), coords(2,:), coords(3,:));
end