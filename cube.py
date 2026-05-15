from random import randint
import tensorflow as tf
import numpy as np

class Cube:
    def __init__(self, ds, init_moves=0):
        '''
        white == 0
        yellow == 1
        orange == 2
        red == 3
        blue == 4
        green == 5
        '''
        self.ds = ds

        if self.ds == 2:
            self.front = [[0] * 2 for i in range(2)]
            self.back = [[1] * 2 for i in range(2)]
            self.left = [[2] * 2 for i in range(2)]
            self.right = [[3] * 2 for i in range(2)]
            self.up = [[4] * 2 for i in range(2)]
            self.bottom = [[5] * 2 for i in range(2)]
            self.last_layers_done = [0, 0]
            self.present_layers_done = [0, 0]
        else:
            self.front = [[0] * 3 for i in range(3)]
            self.back = [[1] * 3 for i in range(3)]
            self.left = [[2] * 3 for i in range(3)]
            self.right = [[3] * 3 for i in range(3)]
            self.up = [[4] * 3 for i in range(3)]
            self.bottom = [[5] * 3 for i in range(3)]
            self.last_layers_done = [0, 0, 0]
            self.present_layers_done = [0, 0, 0]

        self.move_history = []
        self.state_history = []
        #up left front right bottom back
        self.last_walls_done = [0,0,0,0,0,0]
        self.present_walls_done = [0,0,0,0,0,0]
        self.counter = 0

        self.make_random_moves(init_moves)
        self.count_walls_done()
        self.count_layers_done()

    def show(self):
        for i in range(self.ds):
            print("   ",end="")
            for j in range(self.ds):
                print(self.up[i][j], end="")
            print('')
        for i in range(self.ds):
            for j in range(self.ds):
                print(self.left[i][j], end="")
            for j in range(self.ds):
                print(self.front[i][j], end="")
            for j in range(self.ds):
                print(self.right[i][j], end="")
            print('')
        for i in range(self.ds):
            print("   ",end="")
            for j in range(self.ds):
                print(self.bottom[i][j], end="")
            print('')
        for i in range(self.ds):
            print("   ",end="")
            for j in range(self.ds):
                print(self.back[i][j], end="")
            print('')

    def get_state(self):
        return np.array([self.front, self.back, self.left, self.right, self.up, self.bottom]).reshape(1,-1)

    def get_state_one_hot(self):
        return np.array((tf.one_hot(self.front, 6),tf.one_hot(self.back, 6),tf.one_hot(self.left, 6),tf.one_hot(self.right, 6),
                tf.one_hot(self.up, 6),tf.one_hot(self.bottom, 6))).reshape(1,-1)

    def get_move_history(self):
        return self.move_history

    def get_state_history(self):
        return self.state_history

    def count_layers_done(self):
        layers_done = [0,0,0]
        for i in range(self.ds):
            layer = 4
            for side in [self.front, self.back, self.left, self.right]:
                first = side[i][0]
                for j in range(1,len(side)):
                    if side[i][j] == first:
                        layer += 1
                    else:
                        break
                if layer == self.ds*4:
                    layers_done[i] +=1

        self.last_layers_done = self.present_layers_done
        self.present_layers_done = layers_done

    def get_layers_done(self):
        return self.last_layers_done, self.present_layers_done

    def count_walls_done(self):
        walls_done = [0,0,0,0,0,0]
        sides = [self.up, self.left, self.front, self.right, self.bottom, self.back]
        for i in range(6):
            wall = 0
            wall_first = sides[i][0][0]
            for j in range(self.ds):
                for k in range(self.ds):
                    if sides[i][j][k] == wall_first:
                        wall += 1
            if wall == self.ds*self.ds:
                walls_done[i] +=1

        self.last_walls_done = self.present_walls_done
        self.present_walls_done = walls_done

    def get_walls_done(self):
        return self.last_walls_done, self.present_walls_done

    def get_side_incorrect_squares(self, side_name):
        incorrect_squares = 0

        if side_name == 'front':
            side = self.front
        elif side_name == 'back':
            side = self.back
        elif side_name == 'left':
            side = self.left
        elif side_name == 'right':
            side = self.right
        elif side_name == 'up':
            side = self.up
        elif side_name == 'bottom':
            side = self.bottom
        else:
            side = self.front

        if self.ds==2:
            colors = np.array([0,0,0,0,0,0])
            for i in range(2):
                for j in range(2):
                    color = side[i][j]
                    colors[color] += 1

            return 4-colors.max()
        else:
            middle_square = side[1][1]

            for i in range(3):
                for j in range(3):
                    if i==1 and j==1:
                        pass
                    incorrect_squares += int(middle_square != side[i][j])

        return incorrect_squares

    # cube entropy is a sum of all squares on the wrong sides divided by its maximum
    def get_cube_entropy(self):
        total_incorrect_squares = 0

        total_incorrect_squares += self.get_side_incorrect_squares('front')
        total_incorrect_squares += self.get_side_incorrect_squares('back')
        total_incorrect_squares += self.get_side_incorrect_squares('up')
        total_incorrect_squares += self.get_side_incorrect_squares('bottom')
        total_incorrect_squares += self.get_side_incorrect_squares('left')
        total_incorrect_squares += self.get_side_incorrect_squares('right')

        if self.ds==2:
            return total_incorrect_squares/24.0
        else:
            return total_incorrect_squares/48.0

    # every important move on cube
    # 12 moves
    def move_front_clockwise(self):
        if self.ds==2:
            (
                self.front[0][0], self.front[0][1],
                self.front[1][0], self.front[1][1])=(
                self.front[1][0], self.front[0][0],
                self.front[1][1], self.front[0][1]
            )

            (
                self.left[0][1], self.left[1][1],
                self.up[1][0], self.up[1][1],
                self.right[0][0], self.right[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.bottom[0][0], self.bottom[0][1],
                self.left[1][1], self.left[0][1],
                self.up[1][0], self.up[1][1],
                self.right[1][0], self.right[0][0],
            )

        else:
            (
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.front[1][0],                   self.front[1][2],
                self.front[2][0], self.front[2][1], self.front[2][2]) = (

                self.front[2][0], self.front[1][0], self.front[0][0],
                self.front[2][1],                   self.front[0][1],
                self.front[2][2], self.front[1][2], self.front[0][2]
            )


            (
                self.left[0][2], self.left[1][2], self.left[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                ) = (

                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.left[2][2], self.left[1][2], self.left[0][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[2][0], self.right[1][0], self.right[0][0]
            )

    def move_front_counterclockwise(self):
        if self.ds==2:

            (
                self.front[0][0], self.front[0][1],
                self.front[1][0], self.front[1][1])=(
                self.front[0][1], self.front[1][1],
                self.front[0][0], self.front[1][0]
            )

            (
                self.left[0][1], self.left[1][1],
                self.up[1][0], self.up[1][1],
                self.right[0][0], self.right[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (

                self.up[1][1], self.up[1][0],
                self.right[0][0], self.right[1][0],
                self.bottom[0][1], self.bottom[0][0],
                self.left[1][1], self.left[0][1],
            )
        else:
            (
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.front[1][0],                   self.front[1][2],
                self.front[2][0], self.front[2][1], self.front[2][2]) = (

                self.front[0][2], self.front[1][2], self.front[2][2],
                self.front[0][1],                 self.front[2][1],
                self.front[0][0], self.front[1][0], self.front[2][0]
            )

            (
                self.left[0][2], self.left[1][2], self.left[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (
                self.up[2][2], self.up[2][1], self.up[2][0],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][2], self.bottom[0][1], self.bottom[0][0],
                self.left[0][2], self.left[1][2], self.left[2][2],
            )

    def move_front_clockwise_twice(self):
        self.move_front_clockwise()
        self.move_front_clockwise()

    def move_left_clockwise(self):
        if self.ds==2:
            (
                self.left[0][0], self.left[0][1],
                self.left[1][0], self.left[1][1]) = (
                self.left[1][0], self.left[0][0],
                self.left[1][1], self.left[0][1]
            )

            (
                self.back[0][1], self.back[1][1],
                self.up[1][0], self.up[1][1],
                self.front[0][0], self.front[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.bottom[0][0], self.bottom[0][1],
                self.back[1][1], self.back[0][1],
                self.up[1][0], self.up[1][1],
                self.front[1][0], self.front[0][0]
            )
        else:
            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.left[1][0],                  self.left[1][2],
                self.left[2][0], self.left[2][1], self.left[2][2]) = (

                self.left[2][0], self.left[1][0], self.left[0][0],
                self.left[2][1],                  self.left[0][1],
                self.left[2][2], self.left[1][2], self.left[0][2]
            )

            (
                self.back[0][2], self.back[1][2], self.back[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.back[2][2], self.back[1][2], self.back[0][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[2][0], self.front[1][0], self.front[0][0]
            )

    def move_left_counterclockwise(self):
        if self.ds==2:
            (
                self.left[0][0], self.left[0][1],
                self.left[1][0], self.left[1][1]) = (
                self.left[0][1], self.left[1][1],
                self.left[0][0], self.left[1][0]
            )

            (
                self.back[0][1], self.back[1][1],
                self.up[1][0], self.up[1][1],
                self.front[0][0], self.front[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.up[1][1], self.up[1][0],
                self.front[0][0], self.front[1][0],
                self.bottom[0][1], self.bottom[0][0],
                self.back[0][1], self.back[1][1],
            )
        else:
            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.left[1][0],                  self.left[1][2],
                self.left[2][0], self.left[2][1], self.left[2][2]) = (

                self.left[0][2], self.left[1][2], self.left[2][2],
                self.left[0][1],                  self.left[2][1],
                self.left[0][0], self.left[1][0], self.left[2][0]
            )

            (
                self.back[0][2], self.back[1][2], self.back[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (

                self.up[2][2], self.up[2][1], self.up[2][0],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][2], self.bottom[0][1], self.bottom[0][0],
                self.back[0][2], self.back[1][2], self.back[2][2],
            )

    def move_left_clockwise_twice(self):
        self.move_left_clockwise()
        self.move_left_clockwise()

    def move_right_clockwise(self):
        if self.ds==2:
            (
                self.right[0][0], self.right[0][1],
                self.right[1][0], self.right[1][1]) = (
                self.right[1][0], self.right[0][0],
                self.right[1][1], self.right[0][1]
            )

            (
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
            ) = (
                self.bottom[1][0], self.bottom[0][0],
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
            )
        else:
            (
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.right[1][0],                   self.right[1][2],
                self.right[2][0], self.right[2][1], self.right[2][2]) = (

                self.right[2][0], self.right[1][0], self.right[0][0],
                self.right[2][1],                   self.right[0][1],
                self.right[2][2], self.right[1][2], self.right[0][2]
            )

            (
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
            ) = (

                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
            )

    def move_right_counterclockwise(self):
        if self.ds==2:
            (
                self.right[0][0], self.right[0][1],
                self.right[1][0], self.right[1][1]) = (
                self.right[0][1], self.right[1][1],
                self.right[0][0], self.right[1][0]
            )

            (
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
            ) = (
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
                self.front[0][1], self.front[1][1],
            )
        else:

            (
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.right[1][0],                   self.right[1][2],
                self.right[2][0], self.right[2][1], self.right[2][2]) = (

                self.right[0][2], self.right[1][2], self.right[2][2],
                self.right[0][1],                   self.right[2][1],
                self.right[0][0], self.right[1][0], self.right[2][0]
            )

            (
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
            ) = (

                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.front[0][2], self.front[1][2], self.front[2][2],
            )

    def move_right_clockwise_twice(self):
        self.move_right_clockwise()
        self.move_right_clockwise()

    def move_up_clockwise(self):
        if self.ds==2:
            (
                self.up[0][0], self.up[0][1],
                self.up[1][0], self.up[1][1]) = (
                self.up[1][0], self.up[0][0],
                self.up[1][1], self.up[0][1]
            )

            (
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
            ) = (

                self.front[0][0], self.front[0][1],
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
            )
        else:
            (
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.up[1][0],                self.up[1][2],
                self.up[2][0], self.up[2][1], self.up[2][2]) = (

                self.up[2][0], self.up[1][0], self.up[0][0],
                self.up[2][1],                self.up[0][1],
                self.up[2][2], self.up[1][2], self.up[0][2]
            )

            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
            ) = (

                self.front[0][0], self.front[0][1], self.front[0][2],
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
            )

    def move_up_counterclockwise(self):
        if self.ds==2:
            (
                self.up[0][0], self.up[0][1],
                self.up[1][0], self.up[1][1]) = (
                self.up[0][1], self.up[1][1],
                self.up[0][0], self.up[1][0]
            )

            (
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
            ) = (
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
                self.left[0][0], self.left[0][1],
            )
        else:

            (
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.up[1][0],                self.up[1][2],
                self.up[2][0], self.up[2][1], self.up[2][2]) = (

                self.up[0][2], self.up[1][2], self.up[2][2],
                self.up[0][1],                self.up[2][1],
                self.up[0][0], self.up[1][0], self.up[2][0]
            )

            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
            ) = (

                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.left[0][0], self.left[0][1], self.left[0][2],
            )

    def move_up_clockwise_twice(self):
        self.move_up_clockwise()
        self.move_up_clockwise()

    def move_bottom_clockwise(self):
        if self.ds ==2:
            (
                self.bottom[0][0], self.bottom[0][1],
                self.bottom[1][0], self.bottom[1][1]) = (
                self.bottom[1][0], self.bottom[0][0],
                self.bottom[1][1], self.bottom[0][1]
            )

            (
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
            ) = (
                self.front[1][0], self.front[1][1],
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
            )
        else:

            (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.bottom[1][0], self.bottom[1][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2]) = (

                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.bottom[2][1], self.bottom[0][1],
                self.bottom[2][2], self.bottom[1][2], self.bottom[0][2]
            )

            (
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
            ) = (

                self.front[2][0], self.front[2][1], self.front[2][2],
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
            )

    def move_bottom_counterclockwise(self):
        if self.ds==2:
            (
                self.bottom[0][0], self.bottom[0][1],
                self.bottom[1][0], self.bottom[1][1]) = (
                self.bottom[0][1], self.bottom[1][1],
                self.bottom[0][0], self.bottom[1][0]
            )

            (
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
            ) = (
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
                self.left[1][0], self.left[1][1],
            )
        else:

            (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.bottom[1][0], self.bottom[1][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2]) = (

                self.bottom[0][2], self.bottom[1][2], self.bottom[2][2],
                self.bottom[0][1], self.bottom[2][1],
                self.bottom[0][0], self.bottom[1][0], self.bottom[2][0]
            )

            (
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
            ) = (

                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
                self.left[2][0], self.left[2][1], self.left[2][2],
            )

    def move_bottom_clockwise_twice(self):
        self.move_bottom_clockwise()
        self.move_bottom_clockwise()

    def move_back_clockwise(self):
        if self.ds == 2:
            (
                self.back[0][0], self.back[0][1],
                self.back[1][0], self.back[1][1]) = (
                self.back[1][0], self.back[0][0],
                self.back[1][1], self.back[0][1]
            )

            (
                self.left[0][0], self.left[1][0],
                self.up[0][0], self.up[0][1],
                self.right[0][1], self.right[1][1],
                self.bottom[1][0], self.bottom[1][1],
            ) = (
                self.up[0][1], self.up[0][0],
                self.right[0][1], self.right[1][1],
                self.bottom[1][1], self.bottom[1][0],
                self.left[0][0], self.left[1][0],
            )

        else:
            (
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.back[1][0], self.back[1][2],
                self.back[2][0], self.back[2][1], self.back[2][2]) = (

                self.back[2][0], self.back[1][0], self.back[0][0],
                self.back[2][1], self.back[0][1],
                self.back[2][2], self.back[1][2], self.back[0][2]
            )

            (
                self.left[0][0], self.left[1][0], self.left[2][0],
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.right[0][2], self.right[1][2], self.right[2][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2],
            ) = (
                self.up[0][2], self.up[0][1], self.up[0][0],
                self.right[0][2], self.right[1][2], self.right[2][2],
                self.bottom[2][2], self.bottom[2][1], self.bottom[2][0],
                self.left[0][0], self.left[1][0], self.left[2][0]
            )

    def move_back_counterclockwise(self):
        if self.ds == 2:

            (
                self.back[0][0], self.back[0][1],
                self.back[1][0], self.back[1][1]) = (
                self.back[0][1], self.back[1][1],
                self.back[0][0], self.back[1][0]
            )

            (
                self.left[0][0], self.left[1][0],
                self.up[0][0], self.up[0][1],
                self.right[0][1], self.right[1][1],
                self.bottom[1][0], self.bottom[1][1],
            ) = (
                self.bottom[1][0], self.bottom[1][1],
                self.left[1][0], self.left[0][0],
                self.up[0][0], self.up[0][1],
                self.right[1][1], self.right[0][1],
            )
        else:
            (
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.back[1][0], self.back[1][2],
                self.back[2][0], self.back[2][1], self.back[2][2]) = (

                self.back[0][2], self.back[1][2], self.back[2][2],
                self.back[0][1], self.back[2][1],
                self.back[0][0], self.back[1][0], self.back[2][0]
            )

            (
                self.left[0][0], self.left[1][0], self.left[2][0],
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.right[0][2], self.right[1][2], self.right[2][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2],
            ) = (
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2],
                self.left[2][0], self.left[1][0], self.left[0][0],
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.right[2][2], self.right[1][2], self.right[0][2],
            )

    def move_back_clockwise_twice(self):
        self.move_back_clockwise()
        self.move_back_clockwise()

    def make_move(self, move, init=False):
        if move == 0:
            self.move_front_clockwise()
        elif move == 1:
            self.move_front_counterclockwise()
        elif move == 2:
            self.move_left_clockwise()
        elif move == 3:
            self.move_left_counterclockwise()
        elif move == 4:
            self.move_right_clockwise()
        elif move == 5:
            self.move_right_counterclockwise()
        elif move == 6:
            self.move_up_clockwise()
        elif move == 7:
            self.move_up_counterclockwise()
        elif move == 8:
            self.move_bottom_clockwise()
        elif move == 9:
            self.move_bottom_counterclockwise()
        elif move == 10:
            self.move_back_clockwise()
        elif move == 11:
            self.move_back_counterclockwise()
        elif move == 12:
            self.move_front_counterclockwise()
        elif move == 13:
            self.move_left_clockwise_twice()
        elif move == 14:
            self.move_right_clockwise_twice()
        elif move == 15:
            self.move_up_clockwise_twice()
        elif move == 16:
            self.move_bottom_clockwise_twice()
        elif move == 17:
            self.move_back_clockwise_twice()

        if not init:
            self.move_history.append(move)
            self.state_history.append(self.get_state_one_hot())
            self.count_walls_done()
            self.count_layers_done()
            self.counter += 1

        return self.get_state_one_hot(), self.get_cube_entropy(), self.counter, self.get_cube_entropy()==0, self.counter>=100

    def make_random_moves(self, n):
        for i in range(n):
            rand = randint(0,17)
            self.make_move(rand, 1)