"use strict";


let left = (idx) => 2 * idx + 1;
let right = (idx) => left(idx) + 1;
let mid = (low, high) => (high + low) >> 1;
let isEven = (x) => (!(x & 1));
let getRandomInt = (low, high) => low + Math.floor(Math.random() * Math.floor(high - low));
let randomIntegers = (low, high, size) => [...Array(size)].map(_ => getRandomInt(low, high));


let recursive_build_segment_tree = (arr, func) => (function build(T, A, cur, T_left, T_right, func) {
        if (T_left === T_right) {
                T[cur] = A[T_left];
        } else {
                let T_mid = mid(T_left, T_right);
                build(T, A, left(cur), T_left, T_mid, func);
                build(T, A, right(cur), T_mid + 1, T_right, func);
                T[cur] = func(T[left(cur)], T[right(cur)]);
        }
        return T;
})([], arr, 0, 0, arr.length - 1, func);


let iterative_build_segment_tree = function (arr, func) {
        /**
         * @param {Array} arr - an array of totally ordered elements
         * @param {Function} func - a function to be applied to the array of elements
         * @returns {Array} T - a segment tree
         */

        let T = [];
        let n = arr.length;
        for (let i = 0; i < n; i++)
                T[n + i - 1] = arr[i];
        for (let i = n - 2; i >= 0; --i)
                T[i] = func(T[i << 1 | 1], T[(i << 1) + 2]);
        return T;
};

let iterative_query_segment_tree = function (T, l, r, func, identity) {
        /**
         * @param {Number} left -
         * @param {Number} right -
         * Uses a half-open interval
         */
                // length of the SegTree array = 2n - 1
        let n = (T.length + 1) >> 1;
        if (r > n || l < 0) {
                throw new Error("Out of range indices\n");
        }
        let accum = identity;
        for (l += (n - 1), r += (n - 1); l < r; l >>= 1, r >>= 1) {
                // if a node is even, it is a right child of its parent
                if (isEven(l)) accum = func(accum, T[l++]);
                if (isEven(r)) accum = func(accum, T[--r]);
        }
        return accum;
};


let iterative_update_segment_tree = function (T, pos, new_value, func) {
        let n =  ((T.length + 1) >> 1);
        pos += (n - 1); // first leaf starts at index (n - 1)
        T[pos] = func(T[pos], new_value); // update value at the leaf

        // update while walking towards the root
        while (pos > 0) {
                pos >>= 1; // parent position
                T[pos] = func(T[pos << 1 | 1],
                        T[(pos << 1) + 2]);
        }
        T[pos] = func(T[pos << 1 | 1],
                T[(pos << 1) + 2]);
};

let recursive_query_segment_tree = (T, l, r, func, identity, n) =>
        (function query(T, pos, T_left, T_right, l, r, func, identity) {
                        // is completely contained
                        if (l > T_right || r < T_left)
                                return identity;
                        if (l <= T_left && T_right <= r)
                                return T[pos];

                        let T_mid = mid(T_left, T_right);
                        let R_left = query(T, left(pos), T_left, T_mid, l, r, func, identity);
                        let R_right = query(T, right(pos), T_mid + 1, T_right, l, r, func, identity);
                        return func(R_left, R_right);
                }
        )(T, 0, 0, n - 1, l, r, func, identity);


let recursive_update_segment_tree = (T, pos, new_value, func, n) =>
        (function update(T, pos, new_value, func, idx, T_left,
                         T_right) {
                /**
                 * @param {number} pos - index in the array to perform the update
                 * @param {number} n - length of the original array
                 * works by simulating insertion
                 */

                if (T_right === T_left) {
                        T[idx] = func(T[idx], new_value);
                } else {
                        let T_mid = mid(T_left, T_right);
                        if (pos <= T_mid)
                                update(T, pos, new_value, func, left(idx), T_left, T_mid);
                        else
                                update(T, pos, new_value, func, right(idx), T_mid + 1, T_right);

                        T[idx] = func(T[left(idx)], T[right(idx)]);
                }
        })(T, pos, new_value, func, 0, 0, n - 1);


let modify_in_interval = function modify(T, l, r, func, delta, n) {
        /**
         * add a
         */
        for (l += (n - 1), r += (n - 1); l < r; l >>= 1, r >>= 1) {
                if (isEven(l)) T[l] = func(delta, T[l++]);
                if (isEven(r)) T[--r] += func(delta, T[r]);
        }
};

let push_mod_to_leaves = function(T, n, func, identity){
        for (let i = 0; i < (n - 1); ++i) {
                T[left(i)] = func(T[i], T[left(i)]);
                T[right(i)] = func(T[i], T[right(i)]);
                T[i] = identity;
        }
};


let sum = (x, y) => x + y;

let a1 = randomIntegers(0, 30, 10);
let t1 = iterative_build_segment_tree(a1, sum);
let t2 = recursive_build_segment_tree(a1, sum);
console.log(t2.length);
console.log(t1.length);

let r1 = iterative_query_segment_tree(t1, 3, 5, sum, 0);
let r2 = recursive_query_segment_tree(t2, 3, 4, sum, 0, a1.length);
console.log(r1);
console.log(r2);
