#include "wfc.hpp"
#include <limits>

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
