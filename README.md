# andrew_ng_machine_learning

Read these Chapter 1-5 notes: [these notes for more info](https://theptrk.com/2020/02/12/notes-for-coursera-ml-course-week-1-5/)

Q: How to fix this mac error?
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

[1;2;3]~=[1;2;3]

% ans =
% 
%    0
%    0
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

### Get the minimum value and index of a vector
```Octave
[min_value, min_index] = min([10;20;30])

% min_value =  10
% min_index =  1
```

### Get the max value and index of a matrix
This uses `find` as a shortcut
```
minVal = min(min(A));
[row_index, col_index] = find(A == minVal);
```
