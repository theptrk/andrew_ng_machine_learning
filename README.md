# andrew_ng_machine_learning

Read: [these notes for more info](https://theptrk.com/2020/02/12/notes-for-coursera-ml-course-week-1-5/)

Q: How to fix mac error?
```
gnuplot> set terminal aqua enhanced title "Figure 1" ...
```
A: set `GNUTERM` to `X11`
```
> setenv GNUTERM qt
```

## Octave

### Get the error of a vector as a boolean
```Octave 
predictions ~= yval
```
```Octave
[1;2;3]~=[8;8;3]

% ans =
% 
%    1
%    1
%    0
```

### Create a results vector to collect info
```Octave
vector(i) = value
```
```Octave
results=zeros(8,1)

for i=1:8
    results(i) = i * 2
end

% results =
% 
%     2
%     4
%     6
%     8
%    10
```