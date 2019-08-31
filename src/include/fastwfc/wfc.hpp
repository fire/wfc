#ifndef FAST_WFC_WFC_HPP_
#define FAST_WFC_WFC_HPP_

#include <optional>
#include <random>
#include "assert.h"
#include <vector>
#include <tuple>
#include <array>
#include <limits>
#include <vector>
#include <unordered_map>

/**
 * Represent a 2D array.
 * The 2D array is stored in a single array, to improve cache usage.
 */
template <typename T> class Array2D {

public:
	/**
	 * Height and width of the 2D array.
	 */
	unsigned height;
	unsigned width;

	/**
	 * The array containing the data of the 2D array.
	 */
	std::vector<T> data;

	/**
	 * Build a 2D array given its height and width.
	 * All the array elements are initialized to default value.
	 */
	Array2D(unsigned height, unsigned width) noexcept
		: height(height), width(width), data(width * height) {}

	/**
	 * Build a 2D array given its height and width.
	 * All the array elements are initialized to value.
	 */
	Array2D(unsigned height, unsigned width, T value) noexcept
		: height(height), width(width), data(width * height, value) {}

	/**
	 * Return a const reference to the element in the i-th line and j-th column.
	 * i must be lower than height and j lower than width.
	 */
	const T &get(unsigned i, unsigned j) const noexcept {
		assert(i < height && j < width);
		return data[j + i * width];
	}

	/**
	 * Return a reference to the element in the i-th line and j-th column.
	 * i must be lower than height and j lower than width.
	 */
	T &get(unsigned i, unsigned j) noexcept {
		assert(i < height && j < width);
		return data[j + i * width];
	}

	/**
	 * Return the current 2D array reflected along the x axis.
	 */
	Array2D<T> reflected() const noexcept {
		Array2D<T> result = Array2D<T>(width, height);
		for (unsigned y = 0; y < height; y++) {
			for (unsigned x = 0; x < width; x++) {
				result.get(y, x) = get(y, width - 1 - x);
			}
		}
		return result;
	}

	/**
	 * Return the current 2D array rotated 90° anticlockwise
	 */
	Array2D<T> rotated() const noexcept {
		Array2D<T> result = Array2D<T>(width, height);
		for (unsigned y = 0; y < width; y++) {
			for (unsigned x = 0; x < height; x++) {
				result.get(y, x) = get(x, width - 1 - y);
			}
		}
		return result;
	}

	/**
	 * Return the sub 2D array starting from (y,x) and with size (sub_width,
	 * sub_height). The current 2D array is considered toric for this operation.
	 */
	Array2D<T> get_sub_array(unsigned y, unsigned x, unsigned sub_width,
		unsigned sub_height) const noexcept {
		Array2D<T> sub_array_2d = Array2D<T>(sub_width, sub_height);
		for (unsigned ki = 0; ki < sub_height; ki++) {
			for (unsigned kj = 0; kj < sub_width; kj++) {
				sub_array_2d.get(ki, kj) = get((y + ki) % height, (x + kj) % width);
			}
		}
		return sub_array_2d;
	}

	/**
	 * Assign the matrix a to the current matrix.
	 */
	Array2D<T> &operator=(const Array2D<T> &a) noexcept {
		height = a.height;
		width = a.width;
		data = a.data;
		return *this;
	}

	/**
	 * Check if two 2D arrays are equals.
	 */
	bool operator==(const Array2D<T> &a) const noexcept {
		if (height != a.height || width != a.width) {
			return false;
		}

		for (unsigned i = 0; i < data.size(); i++) {
			if (a.data[i] != data[i]) {
				return false;
			}
		}
		return true;
	}
};

/**
 * Hash function.
 */
namespace std {
	template <typename T> class hash<Array2D<T>> {
	public:
		size_t operator()(const Array2D<T> &a) const noexcept {
			std::size_t seed = a.data.size();
			for (const T &i : a.data) {
				seed ^= hash<T>()(i) + (size_t)0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};
} // namespace std


/**
 * A direction is represented by an unsigned integer in the range [0; 3].
 * The x and y values of the direction can be retrieved in these tables.
 */
constexpr int directions_x[4] = { 0, -1, 1, 0 };
constexpr int directions_y[4] = { -1, 0, 0, 1 };

/**
 * Return the opposite direction of direction.
 */
constexpr unsigned get_opposite_direction(unsigned direction) noexcept {
	return 3 - direction;
}

/**
 * Represent a 3D array.
 * The 3D array is stored in a single array, to improve cache usage.
 */
template <typename T> class Array3D {

public:
	/**
	 * The dimensions of the 3D array.
	 */
	unsigned height;
	unsigned width;
	unsigned depth;

	/**
	 * The array containing the data of the 3D array.
	 */
	std::vector<T> data;

	/**
	 * Build a 2D array given its height, width and depth.
	 * All the arrays elements are initialized to default value.
	 */
	Array3D(unsigned height, unsigned width, unsigned depth) noexcept
		: height(height), width(width), depth(depth),
		data(width * height * depth) {}

	/**
	 * Build a 2D array given its height, width and depth.
	 * All the arrays elements are initialized to value
	 */
	Array3D(unsigned height, unsigned width, unsigned depth, T value) noexcept
		: height(height), width(width), depth(depth),
		data(width * height * depth, value) {}

	/**
	 * Return a const reference to the element in the i-th line, j-th column, and
	 * k-th depth. i must be lower than height, j lower than width, and k lower
	 * than depth.
	 */
	const T &get(unsigned i, unsigned j, unsigned k) const noexcept {
		assert(i < height && j < width && k < depth);
		return data[i * width * depth + j * depth + k];
	}

	/**
	 * Return a reference to the element in the i-th line, j-th column, and k-th
	 * depth. i must be lower than height, j lower than width, and k lower than
	 * depth.
	 */
	T &get(unsigned i, unsigned j, unsigned k) noexcept {
		return data[i * width * depth + j * depth + k];
	}

	/**
	 * Check if two 3D arrays are equals.
	 */
	bool operator==(const Array3D &a) const noexcept {
		if (height != a.height || width != a.width || depth != a.depth) {
			return false;
		}

		for (unsigned i = 0; i < data.size(); i++) {
			if (a.data[i] != data[i]) {
				return false;
			}
		}
		return true;
	}
};


class Wave;

/**
 * Propagate information about patterns in the wave.
 */
class Propagator {
public:
	using PropagatorState = std::vector<std::array<std::vector<unsigned>, 4>>;

private:
	/**
	 * The size of the patterns.
	 */
	const unsigned patterns_size;

	/**
	 * propagator[pattern1][direction] contains all the patterns that can
	 * be placed in next to patterns in the direction direction.
	 */
	PropagatorState propagator_state;

	/**
	 * The wave width and height.
	 */
	const unsigned wave_width;
	const unsigned wave_height;

	/**
	 * True if the wave and the output is toric.
	 */
	const bool periodic_output;

	/**
	 * All the tuples (y, x, pattern) that should be propagated.
	 * The tuple should be propagated when wave.get(y, x, pattern) is set to
	 * false.
	 */
	std::vector<std::tuple<unsigned, unsigned, unsigned>> propagating;

	/**
	 * compatible.get(y, x, pattern)[direction] contains the number of patterns
	 * present in the wave that can be placed in the cell next to (y,x) in the
	 * opposite direction of direction without being in contradiction with pattern
	 * placed in (y,x). If wave.get(y, x, pattern) is set to false, then
	 * compatible.get(y, x, pattern) has every element negative or null
	 */
	Array3D<std::array<int, 4>> compatible;

	/**
	 * Initialize compatible.
	 */
	void init_compatible() noexcept;

public:
	/**
	 * Constructor building the propagator and initializing compatible.
	 */
	Propagator(unsigned wave_height, unsigned wave_width, bool periodic_output,
		PropagatorState propagator_state) noexcept
		: patterns_size(propagator_state.size()),
		propagator_state(propagator_state), wave_width(wave_width),
		wave_height(wave_height), periodic_output(periodic_output),
		compatible(wave_height, wave_width, patterns_size) {
		init_compatible();
	}

	/**
	 * Add an element to the propagator.
	 * This function is called when wave.get(y, x, pattern) is set to false.
	 */
	void add_to_propagator(unsigned y, unsigned x, unsigned pattern) noexcept {
		// All the direction are set to 0, since the pattern cannot be set in (y,x).
		std::array<int, 4> temp = {};
		compatible.get(y, x, pattern) = temp;
		propagating.emplace_back(y, x, pattern);
	}

	/**
	 * Propagate the information given with add_to_propagator.
	 */
	void propagate(Wave &wave) noexcept;
};


/**
 * Struct containing the values needed to compute the entropy of all the cells.
 * This struct is updated every time the wave is changed.
 * p'(pattern) is equal to patterns_frequencies[pattern] if wave.get(cell,
 * pattern) is set to true, otherwise 0.
 */
struct EntropyMemoisation {
	std::vector<double> plogp_sum; // The sum of p'(pattern) * log(p'(pattern)).
	std::vector<double> sum;       // The sum of p'(pattern).
	std::vector<double> log_sum;   // The log of sum.
	std::vector<unsigned> nb_patterns; // The number of patterns present
	std::vector<double> entropy;       // The entropy of the cell.
};

/**
 * Contains the pattern possibilities in every cell.
 * Also contains information about cell entropy.
 */
class Wave {
private:
	/**
	 * The patterns frequencies p given to wfc.
	 */
	const std::vector<double> patterns_frequencies;

	/**
	 * The precomputation of p * log(p).
	 */
	const std::vector<double> plogp_patterns_frequencies;

	/**
	 * The precomputation of min (p * log(p)) / 2.
	 * This is used to define the maximum value of the noise.
	 */
	const double min_abs_half_plogp;

	/**
	 * The memoisation of important values for the computation of entropy.
	 */
	EntropyMemoisation memoisation;

	/**
	 * This value is set to true if there is a contradiction in the wave (all
	 * elements set to false in a cell).
	 */
	bool is_impossible;

	/**
	 * The number of distinct patterns.
	 */
	const unsigned nb_patterns;

	/**
	 * The actual wave. data.get(index, pattern) is equal to 0 if the pattern can
	 * be placed in the cell index.
	 */
	Array2D<uint8_t> data;

public:
	/**
	 * The size of the wave.
	 */
	const unsigned width;
	const unsigned height;
	const unsigned size;

	/**
	 * Initialize the wave with every cell being able to have every pattern.
	 */
	Wave(unsigned height, unsigned width,
		const std::vector<double> &patterns_frequencies) noexcept;

	/**
	 * Return true if pattern can be placed in cell index.
	 */
	bool get(unsigned index, unsigned pattern) const noexcept {
		return data.get(index, pattern);
	}

	/**
	 * Return true if pattern can be placed in cell (i,j)
	 */
	bool get(unsigned i, unsigned j, unsigned pattern) const noexcept {
		return get(i * width + j, pattern);
	}

	/**
	 * Set the value of pattern in cell index.
	 */
	void set(unsigned index, unsigned pattern, bool value) noexcept;

	/**
	 * Set the value of pattern in cell (i,j).
	 */
	void set(unsigned i, unsigned j, unsigned pattern, bool value) noexcept {
		set(i * width + j, pattern, value);
	}

	/**
	 * Return the index of the cell with lowest entropy different of 0.
	 * If there is a contradiction in the wave, return -2.
	 * If every cell is decided, return -1.
	 */
	int get_min_entropy(std::minstd_rand &gen) const noexcept;

};

/**
 * Class containing the generic WFC algorithm.
 */
class WFC {
private:
	/**
	 * The random number generator.
	 */
	std::minstd_rand gen;

	/**
	 * The distribution of the patterns as given in input.
	 */
	const std::vector<double> patterns_frequencies;

	/**
	 * The wave, indicating which patterns can be put in which cell.
	 */
	Wave wave;

	/**
	 * The number of distinct patterns.
	 */
	const unsigned nb_patterns;

	/**
	 * The propagator, used to propagate the information in the wave.
	 */
	Propagator propagator;

	/**
	 * Transform the wave to a valid output (a 2d array of patterns that aren't in
	 * contradiction). This function should be used only when all cell of the wave
	 * are defined.
	 */
	Array2D<unsigned> wave_to_output() const noexcept;

public:
	/**
	 * Basic constructor initializing the algorithm.
	 */
	WFC(bool periodic_output, int seed, std::vector<double> patterns_frequencies,
		Propagator::PropagatorState propagator, unsigned wave_height,
		unsigned wave_width)
		noexcept;

	/**
	 * Run the algorithm, and return a result if it succeeded.
	 */
	std::optional<Array2D<unsigned>> run() noexcept;

	/**
	 * Return value of observe.
	 */
	enum ObserveStatus {
		success,    // WFC has finished and has succeeded.
		failure,    // WFC has finished and failed.
		to_continue // WFC isn't finished.
	};

	/**
	 * Define the value of the cell with lowest entropy.
	 */
	ObserveStatus observe() noexcept;

	/**
	 * Propagate the information of the wave.
	 */
	void propagate() noexcept { propagator.propagate(wave); }

	/**
	 * Remove pattern from cell (i,j).
	 */
	void remove_wave_pattern(unsigned i, unsigned j, unsigned pattern) noexcept {
		if (wave.get(i, j, pattern)) {
			wave.set(i, j, pattern, false);
			propagator.add_to_propagator(i, j, pattern);
		}
	}
};

void Propagator::init_compatible() noexcept {
	std::array<int, 4> value;
	// We compute the number of pattern compatible in all directions.
	for (unsigned y = 0; y < wave_height; y++) {
		for (unsigned x = 0; x < wave_width; x++) {
			for (unsigned pattern = 0; pattern < patterns_size; pattern++) {
				for (int direction = 0; direction < 4; direction++) {
					value[direction] =
						propagator_state[pattern][get_opposite_direction(direction)]
						.size();
				}
				compatible.get(y, x, pattern) = value;
			}
		}
	}
}

void Propagator::propagate(Wave &wave) noexcept {

	// We propagate every element while there is element to propagate.
	while (propagating.size() != 0) {

		// The cell and pattern that has been set to false.
		unsigned y1, x1, pattern;
		std::tie(y1, x1, pattern) = propagating.back();
		propagating.pop_back();

		// We propagate the information in all 4 directions.
		for (unsigned direction = 0; direction < 4; direction++) {

			// We get the next cell in the direction direction.
			int dx = directions_x[direction];
			int dy = directions_y[direction];
			int x2, y2;
			if (periodic_output) {
				x2 = ((int)x1 + dx + (int)wave.width) % wave.width;
				y2 = ((int)y1 + dy + (int)wave.height) % wave.height;
			}
			else {
				x2 = x1 + dx;
				y2 = y1 + dy;
				if (x2 < 0 || x2 >= (int)wave.width) {
					continue;
				}
				if (y2 < 0 || y2 >= (int)wave.height) {
					continue;
				}
			}

			// The index of the second cell, and the patterns compatible
			unsigned i2 = x2 + y2 * wave.width;
			const std::vector<unsigned> &patterns =
				propagator_state[pattern][direction];

			// For every pattern that could be placed in that cell without being in
			// contradiction with pattern1
			for (auto it = patterns.begin(), it_end = patterns.end(); it < it_end;
				++it) {

				// We decrease the number of compatible patterns in the opposite
				// direction If the pattern was discarded from the wave, the element
				// is still negative, which is not a problem
				std::array<int, 4> &value = compatible.get(y2, x2, *it);
				value[direction]--;

				// If the element was set to 0 with this operation, we need to remove
				// the pattern from the wave, and propagate the information
				if (value[direction] == 0) {
					add_to_propagator(y2, x2, *it);
					wave.set(i2, *it, false);
				}
			}
		}
	}



}


Array2D<unsigned> WFC::wave_to_output() const noexcept {
	Array2D<unsigned> output_patterns(wave.height, wave.width);
	for (unsigned i = 0; i < wave.size; i++) {
		for (unsigned k = 0; k < nb_patterns; k++) {
			if (wave.get(i, k)) {
				output_patterns.data[i] = k;
			}
		}
	}
	return output_patterns;
}

/**
 * Normalize a vector so the sum of its elements is equal to 1.0f
 */
std::vector<double>& normalize(std::vector<double>& v) {
	double sum_weights = 0.0;
	for (double weight : v) {
		sum_weights += weight;
	}

	double inv_sum_weights = 1.0 / sum_weights;
	for (double& weight : v) {
		weight *= inv_sum_weights;
	}

	return v;
}
/**
 * Return distribution * log(distribution).
 */
std::vector<double>
get_plogp(const std::vector<double> &distribution) noexcept {
	std::vector<double> plogp;
	for (unsigned i = 0; i < distribution.size(); i++) {
		plogp.push_back(distribution[i] * log(distribution[i]));
	}
	return plogp;
}

/**
 * Return min(v) / 2.
 */
double get_min_abs_half(const std::vector<double> &v) noexcept {
	double min_abs_half = std::numeric_limits<double>::infinity();
	for (unsigned i = 0; i < v.size(); i++) {
		min_abs_half = std::min(min_abs_half, std::abs(v[i] / 2.0));
	}
	return min_abs_half;
}

Wave::Wave(unsigned height, unsigned width,
	const std::vector<double> &patterns_frequencies) noexcept
	: patterns_frequencies(patterns_frequencies),
	plogp_patterns_frequencies(get_plogp(patterns_frequencies)),
	min_abs_half_plogp(get_min_abs_half(plogp_patterns_frequencies)),
	is_impossible(false), nb_patterns(patterns_frequencies.size()),
	data(width * height, nb_patterns, 1), width(width), height(height),
	size(height * width) {
	// Initialize the memoisation of entropy.
	double base_entropy = 0;
	double base_s = 0;
	for (unsigned i = 0; i < nb_patterns; i++) {
		base_entropy += plogp_patterns_frequencies[i];
		base_s += patterns_frequencies[i];
	}
	double log_base_s = log(base_s);
	double entropy_base = log_base_s - base_entropy / base_s;
	memoisation.plogp_sum = std::vector<double>(width * height, base_entropy);
	memoisation.sum = std::vector<double>(width * height, base_s);
	memoisation.log_sum = std::vector<double>(width * height, log_base_s);
	memoisation.nb_patterns =
		std::vector<unsigned>(width * height, nb_patterns);
	memoisation.entropy = std::vector<double>(width * height, entropy_base);
}


void Wave::set(unsigned index, unsigned pattern, bool value) noexcept {
	bool old_value = data.get(index, pattern);
	// If the value isn't changed, nothing needs to be done.
	if (old_value == value) {
		return;
	}
	// Otherwise, the memoisation should be updated.
	data.get(index, pattern) = value;
	memoisation.plogp_sum[index] -= plogp_patterns_frequencies[pattern];
	memoisation.sum[index] -= patterns_frequencies[pattern];
	memoisation.log_sum[index] = log(memoisation.sum[index]);
	memoisation.nb_patterns[index]--;
	memoisation.entropy[index] =
		memoisation.log_sum[index] -
		memoisation.plogp_sum[index] / memoisation.sum[index];
	// If there is no patterns possible in the cell, then there is a
	// contradiction.
	if (memoisation.nb_patterns[index] == 0) {
		is_impossible = true;
	}
}


int Wave::get_min_entropy(std::minstd_rand &gen) const noexcept {
	if (is_impossible) {
		return -2;
	}

	std::uniform_real_distribution<> dis(0, min_abs_half_plogp);

	// The minimum entropy (plus a small noise)
	double min = std::numeric_limits<double>::infinity();
	int argmin = -1;

	for (unsigned i = 0; i < size; i++) {

		// If the cell is decided, we do not compute the entropy (which is equal
		// to 0).
		double nb_patterns = memoisation.nb_patterns[i];
		if (nb_patterns == 1) {
			continue;
		}

		// Otherwise, we take the memoised entropy.
		double entropy = memoisation.entropy[i];

		// We first check if the entropy is less than the minimum.
		// This is important to reduce noise computation (which is not
		// negligible).
		if (entropy <= min) {

			// Then, we add noise to decide randomly which will be chosen.
			// noise is smaller than the smallest p * log(p), so the minimum entropy
			// will always be chosen.
			double noise = dis(gen);
			if (entropy + noise < min) {
				min = entropy + noise;
				argmin = i;
			}
		}
	}

	return argmin;
}


WFC::WFC(bool periodic_output, int seed,
	std::vector<double> patterns_frequencies,
	Propagator::PropagatorState propagator, unsigned wave_height,
	unsigned wave_width)
	noexcept
	: gen(seed), patterns_frequencies(normalize(patterns_frequencies)),
	wave(wave_height, wave_width, patterns_frequencies),
	nb_patterns(propagator.size()),
	propagator(wave.height, wave.width, periodic_output, propagator) {}

std::optional<Array2D<unsigned>> WFC::run() noexcept {
	while (true) {

		// Define the value of an undefined cell.
		ObserveStatus result = observe();

		// Check if the algorithm has terminated.
		if (result == failure) {
			return std::nullopt;
		}
		else if (result == success) {
			return wave_to_output();
		}

		// Propagate the information.
		propagator.propagate(wave);
	}
}


WFC::ObserveStatus WFC::observe() noexcept {
	// Get the cell with lowest entropy.
	int argmin = wave.get_min_entropy(gen);

	// If there is a contradiction, the algorithm has failed.
	if (argmin == -2) {
		return failure;
	}

	// If the lowest entropy is 0, then the algorithm has succeeded and
	// finished.
	if (argmin == -1) {
		wave_to_output();
		return success;
	}

	// Choose an element according to the pattern distribution
	double s = 0;
	for (unsigned k = 0; k < nb_patterns; k++) {
		s += wave.get(argmin, k) ? patterns_frequencies[k] : 0;
	}

	std::uniform_real_distribution<> dis(0, s);
	double random_value = dis(gen);
	unsigned chosen_value = nb_patterns - 1;

	for (unsigned k = 0; k < nb_patterns; k++) {
		random_value -= wave.get(argmin, k) ? patterns_frequencies[k] : 0;
		if (random_value <= 0) {
			chosen_value = k;
			break;
		}
	}

	// And define the cell with the pattern.
	for (unsigned k = 0; k < nb_patterns; k++) {
		if (wave.get(argmin, k) != (k == chosen_value)) {
			propagator.add_to_propagator(argmin / wave.width, argmin % wave.width,
				k);
			wave.set(argmin, k, false);
		}
	}

	return to_continue;
}


/**
 * Options needed to use the overlapping wfc.
 */
struct OverlappingWFCOptions {
	bool periodic_input;  // True if the input is toric.
	bool periodic_output; // True if the output is toric.
	unsigned out_height;  // The height of the output in pixels.
	unsigned out_width;   // The width of the output in pixels.
	unsigned symmetry; // The number of symmetries (the order is defined in wfc).
	bool ground;       // True if the ground needs to be set (see init_ground).
	unsigned pattern_size; // The width and height in pixel of the patterns.

	/**
	 * Get the wave height given these options.
	 */
	unsigned get_wave_height() const noexcept {
		return periodic_output ? out_height : out_height - pattern_size + 1;
	}

	/**
	 * Get the wave width given these options.
	 */
	unsigned get_wave_width() const noexcept {
		return periodic_output ? out_width : out_width - pattern_size + 1;
	}
};

/**
 * Class generating a new image with the overlapping WFC algorithm.
 */
template <typename T> class OverlappingWFC {

private:
	/**
	 * The input image. T is usually a color.
	 */
	Array2D<T> input;

	/**
	 * Options needed by the algorithm.
	 */
	OverlappingWFCOptions options;

	/**
	 * The array of the different patterns extracted from the input.
	 */
	std::vector<Array2D<T>> patterns;

	/**
	 * The underlying generic WFC algorithm.
	 */
	WFC wfc;

	/**
	 * Constructor initializing the wfc.
	 * This constructor is called by the other constructors.
	 * This is necessary in order to initialize wfc only once.
	 */
	OverlappingWFC(
		const Array2D<T> &input, const OverlappingWFCOptions &options,
		const int &seed,
		const std::pair<std::vector<Array2D<T>>, std::vector<double>> &patterns,
		const std::vector<std::array<std::vector<unsigned>, 4>>
		&propagator) noexcept
		: input(input), options(options), patterns(patterns.first),
		wfc(options.periodic_output, seed, patterns.second, propagator,
			options.get_wave_height(), options.get_wave_width()) {
		// If necessary, the ground is set.
		if (options.ground) {
			init_ground(wfc, input, patterns.first, options);
		}
	}

	/**
	 * Constructor used only to call the other constructor with more computed
	 * parameters.
	 */
	OverlappingWFC(const Array2D<T> &input, const OverlappingWFCOptions &options,
		const int &seed,
		const std::pair<std::vector<Array2D<T>>, std::vector<double>>
		&patterns) noexcept
		: OverlappingWFC(input, options, seed, patterns,
			generate_compatible(patterns.first)) {}

	/**
	 * Init the ground of the output image.
	 * The lowest middle pattern is used as a floor (and ceiling when the input is
	 * toric) and is placed at the lowest possible pattern position in the output
	 * image, on all its width. The pattern cannot be used at any other place in
	 * the output image.
	 */
	static void init_ground(WFC &wfc, const Array2D<T> &input,
		const std::vector<Array2D<T>> &patterns,
		const OverlappingWFCOptions &options) noexcept {
		unsigned ground_pattern_id =
			get_ground_pattern_id(input, patterns, options);

		// Place the pattern in the ground.
		for (unsigned j = 0; j < options.get_wave_width(); j++) {
			for (unsigned p = 0; p < patterns.size(); p++) {
				if (ground_pattern_id != p) {
					wfc.remove_wave_pattern(options.get_wave_height() - 1, j, p);
				}
			}
		}

		// Remove the pattern from the other positions.
		for (unsigned i = 0; i < options.get_wave_height() - 1; i++) {
			for (unsigned j = 0; j < options.get_wave_width(); j++) {
				wfc.remove_wave_pattern(i, j, ground_pattern_id);
			}
		}

		// Propagate the information with wfc.
		wfc.propagate();
	}

	/**
	 * Return the id of the lowest middle pattern.
	 */
	static unsigned
		get_ground_pattern_id(const Array2D<T> &input,
			const std::vector<Array2D<T>> &patterns,
			const OverlappingWFCOptions &options) noexcept {
		// Get the pattern.
		Array2D<T> ground_pattern =
			input.get_sub_array(input.height - 1, input.width / 2,
				options.pattern_size, options.pattern_size);

		// Retrieve the id of the pattern.
		for (unsigned i = 0; i < patterns.size(); i++) {
			if (ground_pattern == patterns[i]) {
				return i;
			}
		}

		// The pattern exists.
		assert(false);
		return 0;
	}

	/**
	 * Return the list of patterns, as well as their probabilities of apparition.
	 */
	static std::pair<std::vector<Array2D<T>>, std::vector<double>>
		get_patterns(const Array2D<T> &input,
			const OverlappingWFCOptions &options) noexcept {
		std::unordered_map<Array2D<T>, unsigned> patterns_id;
		std::vector<Array2D<T>> patterns;

		// The number of time a pattern is seen in the input image.
		std::vector<double> patterns_weight;

		std::vector<Array2D<T>> symmetries(
			8, Array2D<T>(options.pattern_size, options.pattern_size));
		unsigned max_i = options.periodic_input
			? input.height
			: input.height - options.pattern_size + 1;
		unsigned max_j = options.periodic_input
			? input.width
			: input.width - options.pattern_size + 1;

		for (unsigned i = 0; i < max_i; i++) {
			for (unsigned j = 0; j < max_j; j++) {
				// Compute the symmetries of every pattern in the image.
				symmetries[0].data =
					input
					.get_sub_array(i, j, options.pattern_size, options.pattern_size)
					.data;
				symmetries[1].data = symmetries[0].reflected().data;
				symmetries[2].data = symmetries[0].rotated().data;
				symmetries[3].data = symmetries[2].reflected().data;
				symmetries[4].data = symmetries[2].rotated().data;
				symmetries[5].data = symmetries[4].reflected().data;
				symmetries[6].data = symmetries[4].rotated().data;
				symmetries[7].data = symmetries[6].reflected().data;

				// The number of symmetries in the option class define which symetries
				// will be used.
				for (unsigned k = 0; k < options.symmetry; k++) {
					auto res = patterns_id.insert(
						std::make_pair(symmetries[k], patterns.size()));

					// If the pattern already exist, we just have to increase its number
					// of appearance.
					if (!res.second) {
						patterns_weight[res.first->second] += 1;
					}
					else {
						patterns.push_back(symmetries[k]);
						patterns_weight.push_back(1);
					}
				}
			}
		}

		return { patterns, patterns_weight };
	}

	/**
	 * Return true if the pattern1 is compatible with pattern2
	 * when pattern2 is at a distance (dy,dx) from pattern1.
	 */
	static bool agrees(const Array2D<T> &pattern1, const Array2D<T> &pattern2,
		int dy, int dx) noexcept {
		unsigned xmin = dx < 0 ? 0 : dx;
		unsigned xmax = dx < 0 ? dx + pattern2.width : pattern1.width;
		unsigned ymin = dy < 0 ? 0 : dy;
		unsigned ymax = dy < 0 ? dy + pattern2.height : pattern1.width;

		// Iterate on every pixel contained in the intersection of the two pattern.
		for (unsigned y = ymin; y < ymax; y++) {
			for (unsigned x = xmin; x < xmax; x++) {
				// Check if the color is the same in the two patterns in that pixel.
				if (pattern1.get(y, x) != pattern2.get(y - dy, x - dx)) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Precompute the function agrees(pattern1, pattern2, dy, dx).
	 * If agrees(pattern1, pattern2, dy, dx), then compatible[pattern1][direction]
	 * contains pattern2, where direction is the direction defined by (dy, dx)
	 * (see direction.hpp).
	 */
	static std::vector<std::array<std::vector<unsigned>, 4>>
		generate_compatible(const std::vector<Array2D<T>> &patterns) noexcept {
		std::vector<std::array<std::vector<unsigned>, 4>> compatible =
			std::vector<std::array<std::vector<unsigned>, 4>>(patterns.size());

		// Iterate on every dy, dx, pattern1 and pattern2
		for (unsigned pattern1 = 0; pattern1 < patterns.size(); pattern1++) {
			for (unsigned direction = 0; direction < 4; direction++) {
				for (unsigned pattern2 = 0; pattern2 < patterns.size(); pattern2++) {
					if (agrees(patterns[pattern1], patterns[pattern2],
						directions_y[direction], directions_x[direction])) {
						compatible[pattern1][direction].push_back(pattern2);
					}
				}
			}
		}

		return compatible;
	}

	/**
	 * Transform a 2D array containing the patterns id to a 2D array containing
	 * the pixels.
	 */
	Array2D<T> to_image(const Array2D<unsigned> &output_patterns) const noexcept {
		Array2D<T> output = Array2D<T>(options.out_height, options.out_width);

		if (options.periodic_output) {
			for (unsigned y = 0; y < options.get_wave_height(); y++) {
				for (unsigned x = 0; x < options.get_wave_width(); x++) {
					output.get(y, x) = patterns[output_patterns.get(y, x)].get(0, 0);
				}
			}
		}
		else {
			for (unsigned y = 0; y < options.get_wave_height(); y++) {
				for (unsigned x = 0; x < options.get_wave_width(); x++) {
					output.get(y, x) = patterns[output_patterns.get(y, x)].get(0, 0);
				}
			}
			for (unsigned y = 0; y < options.get_wave_height(); y++) {
				const Array2D<T> &pattern =
					patterns[output_patterns.get(y, options.get_wave_width() - 1)];
				for (unsigned dx = 1; dx < options.pattern_size; dx++) {
					output.get(y, options.get_wave_width() - 1 + dx) = pattern.get(0, dx);
				}
			}
			for (unsigned x = 0; x < options.get_wave_width(); x++) {
				const Array2D<T> &pattern =
					patterns[output_patterns.get(options.get_wave_height() - 1, x)];
				for (unsigned dy = 1; dy < options.pattern_size; dy++) {
					output.get(options.get_wave_height() - 1 + dy, x) =
						pattern.get(dy, 0);
				}
			}
			const Array2D<T> &pattern = patterns[output_patterns.get(
				options.get_wave_height() - 1, options.get_wave_width() - 1)];
			for (unsigned dy = 1; dy < options.pattern_size; dy++) {
				for (unsigned dx = 1; dx < options.pattern_size; dx++) {
					output.get(options.get_wave_height() - 1 + dy,
						options.get_wave_width() - 1 + dx) = pattern.get(dy, dx);
				}
			}
		}

		return output;
	}

public:
	/**
	 * The constructor used by the user.
	 */
	OverlappingWFC(const Array2D<T> &input, const OverlappingWFCOptions &options,
		int seed) noexcept
		: OverlappingWFC(input, options, seed, get_patterns(input, options)) {}

	/**
	 * Run the WFC algorithm, and return the result if the algorithm succeeded.
	 */
	std::optional<Array2D<T>> run() noexcept {
		std::optional<Array2D<unsigned>> result = wfc.run();
		if (result.has_value()) {
			return to_image(*result);
		}
		return std::nullopt;
	}
};


/**
 * The distinct symmetries of a tile.
 * It represents how the tile behave when it is rotated or reflected
 */
enum class Symmetry { X, T, I, L, backslash, P };

/**
 * Return the number of possible distinct orientations for a tile.
 * An orientation is a combination of rotations and reflections.
 */
unsigned nb_of_possible_orientations(const Symmetry &symmetry) {
	switch (symmetry) {
	case Symmetry::X:
		return 1;
	case Symmetry::I:
	case Symmetry::backslash:
		return 2;
	case Symmetry::T:
	case Symmetry::L:
		return 4;
	default:
		return 8;
	}
}

/**
 * A tile that can be placed on the board.
 */
template <typename T> struct Tile {
	std::vector<Array2D<T>> data; // The different orientations of the tile
	Symmetry symmetry;            // The symmetry of the tile
	double weight; // Its weight on the distribution of presence of tiles

	/**
	 * Generate the map associating an orientation id to the orientation
	 * id obtained when rotating 90° anticlockwise the tile.
	 */
	static std::vector<unsigned>
		generate_rotation_map(const Symmetry &symmetry) noexcept {
		switch (symmetry) {
		case Symmetry::X:
			return { 0 };
		case Symmetry::I:
		case Symmetry::backslash:
			return { 1, 0 };
		case Symmetry::T:
		case Symmetry::L:
			return { 1, 2, 3, 0 };
		case Symmetry::P:
		default:
			return { 1, 2, 3, 0, 5, 6, 7, 4 };
		}
	}

	/**
	 * Generate the map associating an orientation id to the orientation
	 * id obtained when reflecting the tile along the x axis.
	 */
	static std::vector<unsigned>
		generate_reflection_map(const Symmetry &symmetry) noexcept {
		switch (symmetry) {
		case Symmetry::X:
			return { 0 };
		case Symmetry::I:
			return { 0, 1 };
		case Symmetry::backslash:
			return { 1, 0 };
		case Symmetry::T:
			return { 0, 3, 2, 1 };
		case Symmetry::L:
			return { 1, 0, 3, 2 };
		case Symmetry::P:
		default:
			return { 4, 7, 6, 5, 0, 3, 2, 1 };
		}
	}

	/**
	 * Generate the map associating an orientation id and an action to the
	 * resulting orientation id.
	 * Actions 0, 1, 2, and 3 are 0°, 90°, 180°, and 270° anticlockwise rotations.
	 * Actions 4, 5, 6, and 7 are actions 0, 1, 2, and 3 preceded by a reflection
	 * on the x axis.
	 */
	static std::vector<std::vector<unsigned>>
		generate_action_map(const Symmetry &symmetry) noexcept {
		std::vector<unsigned> rotation_map = generate_rotation_map(symmetry);
		std::vector<unsigned> reflection_map = generate_reflection_map(symmetry);
		size_t size = rotation_map.size();
		std::vector<std::vector<unsigned>> action_map(8,
			std::vector<unsigned>(size));
		for (size_t i = 0; i < size; ++i) {
			action_map[0][i] = i;
		}

		for (size_t a = 1; a < 4; ++a) {
			for (size_t i = 0; i < size; ++i) {
				action_map[a][i] = rotation_map[action_map[a - 1][i]];
			}
		}
		for (size_t i = 0; i < size; ++i) {
			action_map[4][i] = reflection_map[action_map[0][i]];
		}
		for (size_t a = 5; a < 8; ++a) {
			for (size_t i = 0; i < size; ++i) {
				action_map[a][i] = rotation_map[action_map[a - 1][i]];
			}
		}
		return action_map;
	}

	/**
	 * Generate all distincts rotations of a 2D array given its symmetries;
	 */
	static std::vector<Array2D<T>> generate_oriented(Array2D<T> data,
		Symmetry symmetry) noexcept {
		std::vector<Array2D<T>> oriented;
		oriented.push_back(data);

		switch (symmetry) {
		case Symmetry::I:
		case Symmetry::backslash:
			oriented.push_back(data.rotated());
			break;
		case Symmetry::T:
		case Symmetry::L:
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			break;
		case Symmetry::P:
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated().reflected());
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			oriented.push_back(data = data.rotated());
			break;
		default:
			break;
		}

		return oriented;
	}

	/**
	 * Create a tile with its differents orientations, its symmetries and its
	 * weight on the distribution of tiles.
	 */
	Tile(std::vector<Array2D<T>> data, Symmetry symmetry, double weight) noexcept
		: data(data), symmetry(symmetry), weight(weight) {}

	/*
	 * Create a tile with its base orientation, its symmetries and its
	 * weight on the distribution of tiles.
	 * The other orientations are generated with its first one.
	 */
	Tile(Array2D<T> data, Symmetry symmetry, double weight) noexcept
		: data(generate_oriented(data, symmetry)), symmetry(symmetry),
		weight(weight) {}
};

/**
 * Options needed to use the tiling wfc.
 */
struct TilingWFCOptions {
	bool periodic_output;
};

/**
 * Class generating a new image with the tiling WFC algorithm.
 */
template <typename T> class TilingWFC {
private:
	/**
	 * The distincts tiles.
	 */
	std::vector<Tile<T>> tiles;

	/**
	 * Map ids of oriented tiles to tile and orientation.
	 */
	std::vector<std::pair<unsigned, unsigned>> id_to_oriented_tile;

	/**
	 * Map tile and orientation to oriented tile id.
	 */
	std::vector<std::vector<unsigned>> oriented_tile_ids;

	/**
	 * Otions needed to use the tiling wfc.
	 */
	TilingWFCOptions options;

	/**
	 * The underlying generic WFC algorithm.
	 */
	WFC wfc;

	/**
	 * Generate mapping from id to oriented tiles and vice versa.
	 */
	static std::pair<std::vector<std::pair<unsigned, unsigned>>,
		std::vector<std::vector<unsigned>>>
		generate_oriented_tile_ids(const std::vector<Tile<T>> &tiles) noexcept {
		std::vector<std::pair<unsigned, unsigned>> id_to_oriented_tile;
		std::vector<std::vector<unsigned>> oriented_tile_ids;

		unsigned id = 0;
		for (unsigned i = 0; i < tiles.size(); i++) {
			oriented_tile_ids.push_back({});
			for (unsigned j = 0; j < tiles[i].data.size(); j++) {
				id_to_oriented_tile.push_back({ i, j });
				oriented_tile_ids[i].push_back(id);
				id++;
			}
		}

		return { id_to_oriented_tile, oriented_tile_ids };
	}

	/**
	 * Generate the propagator which will be used in the wfc algorithm.
	 */
	static std::vector<std::array<std::vector<unsigned>, 4>> generate_propagator(
		const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>>
		&neighbors,
		std::vector<Tile<T>> tiles,
		std::vector<std::pair<unsigned, unsigned>> id_to_oriented_tile,
		std::vector<std::vector<unsigned>> oriented_tile_ids) {
		size_t nb_oriented_tiles = id_to_oriented_tile.size();
		std::vector<std::array<std::vector<bool>, 4>> dense_propagator(
			nb_oriented_tiles, { std::vector<bool>(nb_oriented_tiles, false),
								std::vector<bool>(nb_oriented_tiles, false),
								std::vector<bool>(nb_oriented_tiles, false),
								std::vector<bool>(nb_oriented_tiles, false) });

		for (auto neighbor : neighbors) {
			unsigned tile1 = std::get<0>(neighbor);
			unsigned orientation1 = std::get<1>(neighbor);
			unsigned tile2 = std::get<2>(neighbor);
			unsigned orientation2 = std::get<3>(neighbor);
			std::vector<std::vector<unsigned>> action_map1 =
				Tile<T>::generate_action_map(tiles[tile1].symmetry);
			std::vector<std::vector<unsigned>> action_map2 =
				Tile<T>::generate_action_map(tiles[tile2].symmetry);

			auto add = [&](unsigned action, unsigned direction) {
				unsigned temp_orientation1 = action_map1[action][orientation1];
				unsigned temp_orientation2 = action_map2[action][orientation2];
				unsigned oriented_tile_id1 =
					oriented_tile_ids[tile1][temp_orientation1];
				unsigned oriented_tile_id2 =
					oriented_tile_ids[tile2][temp_orientation2];
				dense_propagator[oriented_tile_id1][direction][oriented_tile_id2] =
					true;
				direction = get_opposite_direction(direction);
				dense_propagator[oriented_tile_id2][direction][oriented_tile_id1] =
					true;
			};

			add(0, 2);
			add(1, 0);
			add(2, 1);
			add(3, 3);
			add(4, 1);
			add(5, 3);
			add(6, 2);
			add(7, 0);
		}

		std::vector<std::array<std::vector<unsigned>, 4>> propagator(
			nb_oriented_tiles);
		for (size_t i = 0; i < nb_oriented_tiles; ++i) {
			for (size_t j = 0; j < nb_oriented_tiles; ++j) {
				for (size_t d = 0; d < 4; ++d) {
					if (dense_propagator[i][d][j]) {
						propagator[i][d].push_back(j);
					}
				}
			}
		}

		return propagator;
	}

	/**
	 * Get probability of presence of tiles.
	 */
	static std::vector<double>
		get_tiles_weights(const std::vector<Tile<T>> &tiles) {
		std::vector<double> frequencies;
		for (size_t i = 0; i < tiles.size(); ++i) {
			for (size_t j = 0; j < tiles[i].data.size(); ++j) {
				frequencies.push_back(tiles[i].weight / tiles[i].data.size());
			}
		}
		return frequencies;
	}

	/**
	 * Translate the generic WFC result into the image result
	 */
	Array2D<T> id_to_tiling(Array2D<unsigned> ids) {
		unsigned size = tiles[0].data[0].height;
		Array2D<T> tiling(size * ids.height, size * ids.width);
		for (unsigned i = 0; i < ids.height; i++) {
			for (unsigned j = 0; j < ids.width; j++) {
				std::pair<unsigned, unsigned> oriented_tile =
					id_to_oriented_tile[ids.get(i, j)];
				for (unsigned y = 0; y < size; y++) {
					for (unsigned x = 0; x < size; x++) {
						tiling.get(i * size + y, j * size + x) =
							tiles[oriented_tile.first].data[oriented_tile.second].get(y, x);
					}
				}
			}
		}
		return tiling;
	}

public:
	/**
	 * Construct the TilingWFC class to generate a tiled image.
	 */
	TilingWFC(
		const std::vector<Tile<T>> &tiles,
		const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>>
		&neighbors,
		const unsigned height, const unsigned width,
		const TilingWFCOptions &options, int seed)
		: tiles(tiles),
		id_to_oriented_tile(generate_oriented_tile_ids(tiles).first),
		oriented_tile_ids(generate_oriented_tile_ids(tiles).second),
		options(options),
		wfc(options.periodic_output, seed, get_tiles_weights(tiles),
			generate_propagator(neighbors, tiles, id_to_oriented_tile,
				oriented_tile_ids),
			height, width) {}

	/**
	 * Run the tiling wfc and return the result if the algorithm succeeded
	 */
	std::optional<Array2D<T>> run() {
		auto a = wfc.run();
		if (a == std::nullopt) {
			return std::nullopt;
		}
		return id_to_tiling(*a);
	}
};
#endif // FAST_WFC_WFC_HPP_
