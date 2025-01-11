'''
A 4x4 arena with walls everywhere.
V %empty initial set of visited states.
s random starting location.
Define function to walk through the maze and remove walls
Function walk maze(s,A,V)
V.add(s)%Add s to set of visited states
Nneighbors(s)%Neighbors of s,including those through the periodic boundaries
Iterate through all neighboring states in random order
forn∈randomize(W)do
If we reached a state we have not seen before
if ny then
A.remove_wall(s,n)%Remove wall between s and n from arena
.=walk maze(n.A.1%Contine from now state
return A,V
A,V=walk_maze(s,A,V)%Construct maze using our recursive algorithm
%Remove 3 additional walls at random to increase the degeneracy of the tasks.
%This increases the number of decision points with multiple routes to the goal.
for i=1:3 do
w random_wall(A)%Select one of the remaining walls at random
A.remove__wall(w)%Remove from set of walls
return A Return the maze we constructed
'''
import numpy as np

def neighbors(maze_size, start, dirs, wrap):
    neighbors = []
    for d in dirs:
        n = start + d
        if wrap:
            n = n % maze_size
        if 0 <= n[0] < maze_size and 0 <= n[1] < maze_size:
            neighbors.append(n)
    return neighbors

def dir_to_idx(d):
    dirs = np.array([[-1, 0], [0, 1], [1, 0], [-1, 0]])
    for (i, dir) in enumerate(dirs):
        if np.all(dir == d):
            return i
        
    return -1

def walk(maze_size, maz, start, dirs, visited: set, wrap: bool):
    visited.add(start[0] * maze_size + start[1])
    for d in np.random.permutation(dirs):
        n = start + d
        if wrap:
            n = n % maze_size
        if 0 <= n[0] < maze_size and 0 <= n[1] < maze_size:
            n_idx = n[0] * maze_size + n[1]
            if n_idx not in visited:
                maz[start[0], start[1], dir_to_idx(d=d)] = 0
                maz[n_idx // maze_size, n_idx % maze_size, ((dir_to_idx(d=d) + 2) % maze_size)] = 0
                walk(maze_size, maz, n, dirs, visited, wrap)

def maze(maze_size, wrap = True):
    dirs = np.array([[-1, 0], [0, 1], [1, 0], [-1, 0]])
    maz = np.ones((maze_size, maze_size, 4), dtype=np.float32)
    start = np.random.randint(0, maze_size, size=2)
    visited = set()
    
    walk(maze_size, maz, start, dirs, visited, wrap)
    
    # remove a couple of additional walls to increase degeneracy
    if wrap:
        holes = 3 * (maze_size - 3) #3 for Larena=4, 6 for Larena = 5
    else:
        holes = 4 * (maze_size - 3) #4 for Larena=4, 8 for Larena = 5
        msize = maze_size - 1
        # note permanent walls
        maz[msize, :, 2] = 0.5; maz[0, :, 0] = 0.5
        maz[:, msize, 1] = 0.5; maz[:, 0, 3] = 0.5
        
    for _ in range(holes):
        walls = np.argwhere(maz == 1)
        idx = np.random.choice(len(walls))
        wall = walls[idx]
        start = wall[:2]
        d = wall[2]
        n = start + dirs[d]
        n = n % maze_size
        maz[start[0], start[1], d] = 0
        maz[n[0], n[1], (d + 2) % maze_size] = 0

    maz[maz == 0.5] = 1
    
    # 交换第一个和第二个维度
    maz = np.transpose(maz, (1, 0, 2))

    # 重塑为二维数组
    maz = maz.reshape(maze_size * maze_size, 4)
    return maz.astype(np.int32)

    
if __name__ == "__main__":
    maze(4)