"use strict";

/** Implementation of a segment tree.
 * @author Erastus <erastusmurungi@gmail.com>
 */

let left = (idx) => idx << 1 | 1;
let right = (idx) => (idx << 1) + 2;
let mid = (low, high) => (high + low) >> 1;
let isEven = (x) => (!(x & 1));
let getRandomInt = (low, high) => low + Math.floor(Math.random() * Math.floor(high - low));
let randomIntegers = (low, high, size) => [...Array(size)].map(_ => getRandomInt(low, high));


let recursive_build_segment_tree = (arr, func) => (function build(seg, A, cur, t_left, t_right, func) {
        if (t_left === t_right) {
                seg[cur] = A[t_left];
        } else {
                let t_mid = mid(t_left, t_right);
                build(seg, A, left(cur), t_left, t_mid, func);
                build(seg, A, right(cur), t_mid + 1, t_right, func);
                seg[cur] = func(seg[left(cur)], seg[right(cur)]);
        }
        return seg;
})([], arr, 0, 0, arr.length - 1, func);


let iterative_build_segment_tree = function (arr, func) {
        /**
         * @param {Array} arr - an array of totally ordered elements
         * @param {Function} func - a function to be applied to the array of elements
         * @returns {Array} seg - a segment tree
         */

        let seg = [];
        let n = arr.length;
        for (let i = 0; i < n; i++)
                seg[n + i - 1] = arr[i];
        for (let i = n - 2; i >= 0; --i)
                seg[i] = func(seg[i << 1 | 1], seg[(i << 1) + 2]);
        return seg;
};

let iterative_query_segment_tree = function (seg, l, r, func, identity) {
        /**
         * @param {Number} left - left endpoint of the query
         * @param {Number} right - right endpoint of the query
         * Uses a half-open interval
         */

        // length of the SegTree array = 2n - 1 ; n = ((2n - 1) + 1) / 2
        let n = (seg.length + 1) >> 1;
        if (r > n || l < 0) {
                throw new Error("Out of range indices\n");
        }
        let accum = identity;
        for (l += (n - 1), r += (n - 1); l < r; l >>= 1, r >>= 1) {
                // if a node is even, it is a right child of its parent
                if (isEven(l)) accum = func(accum, seg[l++]);
                if (isEven(r)) accum = func(accum, seg[--r]);
        }
        return accum;
};


let iterative_update_segment_tree = function (seg, pos, new_value, func) {
        let n = ((seg.length + 1) >> 1);
        pos += (n - 1); // first leaf starts at index (n - 1)
        seg[pos] = func(seg[pos], new_value); // update value at the leaf

        // update while walking towards the root
        while (pos > 0) {
                pos >>= 1; // parent position
                seg[pos] = func(seg[pos << 1 | 1],
                        seg[(pos << 1) + 2]);
        }
        seg[pos] = func(seg[pos << 1 | 1],
                seg[(pos << 1) + 2]);
};

let recursive_query_segment_tree = (seg, l, r, func, identity, n) =>
        (function query(seg, pos, t_left, t_right, l, r, func, identity) {
                        if (l > t_right || r < t_left)
                                return identity;
                        if (l <= t_left && t_right <= r)
                                return seg[pos];

                        let t_mid = mid(t_left, t_right);
                        let r_left = query(seg, left(pos), t_left, t_mid, l, r, func, identity);
                        let r_right = query(seg, right(pos), t_mid + 1, t_right, l, r, func, identity);
                        return func(r_left, r_right);
                }
        )(seg, 0, 0, n - 1, l, r, func, identity);


let recursive_update_segment_tree = (seg, pos, new_value, func, n) =>
        (function update(seg, pos, new_value, func, idx, t_left,
                         t_right) {
                /**
                 * @param {Number} pos - index in the array to perform the update
                 * @param {Number} n - length of the original array
                 * works by simulating insertion
                 */

                if (t_right === t_left) {
                        seg[idx] = func(seg[idx], new_value);
                } else {
                        let t_mid = mid(t_left, t_right);
                        if (pos <= t_mid)
                                update(seg, pos, new_value, func, left(idx), t_left, t_mid);
                        else
                                update(seg, pos, new_value, func, right(idx), t_mid + 1, t_right);

                        seg[idx] = func(seg[left(idx)], seg[right(idx)]);
                }
        })(seg, pos, new_value, func, 0, 0, n - 1);


let iterative_update_interval = (seg, l, r, func, delta, n) => {
        /**
         * update all the nodes in an interval
         */
        for (l += (n - 1), r += (n - 1); l < r; l >>= 1, r >>= 1) {
                if (isEven(l)) seg[l] = func(delta, seg[l++]);
                if (isEven(r)) seg[--r] = func(delta, seg[r]);
        }
};


let recursive_update_interval = (seg, l, r, func, delta, n) =>
        (function update_interval(seg, cur, t_left, t_right, l, r, func, delta) {
                if (l > t_right || r < t_left)
                        return;
                if (l <= t_left && t_right <= r)
                        return (seg[cur] = func(seg[cur], delta));


                let t_mid = mid(t_left, t_right);
                update_interval(seg, left(cur), t_left, t_mid, l, r, func, delta);
                update_interval(seg, right(cur), t_mid + 1, t_right, l, r, func, delta);
                seg[cur] = func(seg[left(cur)], seg[right(cur)]);

        })(seg, 0, 0, n - 1, l, r, func, delta);

let push_mod_to_leaves = function (seg, n, func, identity) {
        for (let i = 0; i < (n - 1); ++i) {
                seg[left(i)] = func(seg[i], seg[left(i)]);
                seg[right(i)] = func(seg[i], seg[right(i)]);
                seg[i] = identity;
        }
};

let recursive_lazy_update_interval = (seg, lazy, l, r, func, identity, delta, n) =>
        (function update_interval_lazy(seg, lazy, cur, t_left, t_right, l, r, func, identity, delta) {
                if (lazy[cur] !== identity) {
                        seg[cur] = func(seg[cur], lazy[cur]);
                        // is not a leaf
                        if (t_left !== t_right) {
                                lazy[left(cur)] = func(lazy[left(cur)], lazy[cur]);
                                // push the changes to the children
                                lazy[right(cur)] = func(lazy[right(cur)], lazy[cur]);
                        }
                        lazy[cur] = identity;
                }
                // no overlap
                if (l > t_right || r < t_left)
                        return;
                // total overlap
                if (l <= t_left && t_right <= r) {
                        seg[cur] = func(seg[cur], delta);
                        // instead of going all the way to the leaves, we only update seg[cur].left and seg[cur].right
                        // and return
                        if (t_left !== t_right) {
                                lazy[left(cur)] = func(lazy[left(cur)], delta);
                                lazy[right(cur)] = func(lazy[right(cur)], delta);
                        }
                        return;
                }

                let t_mid = mid(t_left, t_right);
                update_interval_lazy(seg, left(cur), t_left, t_mid, l, r, func, identity, delta);
                update_interval_lazy(seg, right(cur), t_mid + 1, t_right, l, r, func, identity, delta);
                seg[cur] = func(seg[left(cur)], seg[right(cur)]);

        })(seg, 0, 0, n - 1, l, r, func, identity, delta);


let recursive_query_segment_tree_lazy = (seg, lazy, l, r, func, identity, n) =>
        (function query_lazy(seg, cur, lazy, t_left, t_right, l, r, func, identity) {
                if (l > t_right || r < t_left)
                        return identity;
                if (lazy[cur] !== identity) {
                        seg[cur] = func(seg[cur], lazy[cur]);
                        if (t_left !== t_right) {
                                lazy[left(cur)] = func(lazy[left(cur)], lazy[cur]);
                                lazy[right(cur)] = func(lazy[right(cur)], lazy[cur]);
                        }
                        lazy[cur] = identity;
                }
                if (l <= t_left && t_right <= r)
                        return seg[cur];

                let t_mid = mid(t_left, t_right);
                let r_left = query_lazy(seg, left(cur), t_left, t_mid, l, r, func, identity);
                let r_right = query_lazy(seg, right(cur), t_mid + 1, t_right, l, r, func, identity);
                return func(r_left, r_right);
        })(seg, 0, lazy, 0, n - 1, l, r, func, identity);

let seg_node = function (val, lc, rc) {
        this.lc = lc;
        this.rc = rc;
        this.val = val;
};

let build_persistent_segment_tree = (arr, func) => {
        let roots = [];
        roots.push((function build_persistent(node, A, t_left, t_right, func) {
                if (t_left === t_right) {
                        node.val = A[t_left];
                } else {
                        let t_mid = mid(t_left, t_right);
                        node.lc = new seg_node();
                        node.rc = new seg_node();
                        build_persistent(node.lc, A, t_left, t_mid, func);
                        build_persistent(node.rc, A, t_mid + 1, t_right, func);
                        node.val = func(node.lc.val, node.rc.val);
                }
                return node;
        })(new seg_node(), arr, 0, arr.length - 1, func));
        return roots;
};


let update_persistent = function (roots, p, func, new_value, n) {
        let prev_root = roots[roots.length - 1];
        let curr_root = new seg_node();
        (function update(prev_node, cur_node, t_left, t_right, p, func, new_value) {
                if (p > t_right || p < t_left || t_left > t_right)
                        return;
                // leaf node
                if (t_left === t_right)
                        return cur_node.val = func(prev_node.val, new_value);

                let t_mid = mid(t_left, t_right);

                if (p <= t_mid) {
                        cur_node.rc = prev_node.rc;
                        cur_node.lc = new seg_node();
                        update(prev_node.lc, cur_node.lc, t_left, t_mid, p, func, new_value);
                } else {
                        cur_node.lc = prev_node.lc;
                        cur_node.rc = new seg_node();
                        update(prev_node.rc, cur_node.rc, ++t_mid, t_right, p, func, new_value);
                }
                cur_node.val = func(cur_node.lc.val, cur_node.rc.val);
        })(prev_root, curr_root, 0, n - 1, p, func, new_value);
        roots.push(curr_root);
};

let query_persistent = (roots, time, l, r, func, identity, n) => {
        if (time >= roots.length) {
                throw new Error("we are not there yet");
        }
        return (function query(cur_node, t_left, t_right, l, r, func, identity) {
                if (l > t_right || r < t_left || t_left > t_right)
                        return identity;
                if (l <= t_left && t_right <= r)
                        return cur_node.val;
                let t_mid = mid(t_left, t_right);
                let r_left = query(cur_node.lc, t_left, t_mid, l, r, func, identity);
                let r_right = query(cur_node.rc, t_mid + 1, t_right, l, r, func, identity);
                return func(r_left, r_right);
        })(roots[time], 0, n - 1, l, r, func, identity);
};

let sum = (x, y) => x + y;
let sum_id = 0;

let a1 = randomIntegers(0, 30, 6);
// let t1 = iterative_build_segment_tree(a1, sum);
// let t2 = recursive_build_segment_tree(a1, sum);

// let r1 = iterative_query_segment_tree(t1, 3, 5, sum, 0);
// let r2 = recursive_query_segment_tree(t2, 3, 4, sum, 0, a1.length);
// console.log(r1);
// console.log(r2);
console.log(a1);

let p1 = build_persistent_segment_tree(a1, sum);
update_persistent(p1, 2, sum, 10, a1.length);

console.log(p1);
let r3 = query_persistent(p1, 0, 2, 3, sum, sum_id, a1.length);

let r4 = query_persistent(p1, 1, 2, 3, sum, sum_id, a1.length);
console.log(r3);
console.log(r4);
