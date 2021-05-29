#ifndef INITIALIZEROPTIONS_H
#define INITIALIZEROPTIONS_H

namespace orcvio {

    /**
     * @brief Struct which stores all our feature initializer options
     */
    struct FeatureInitializerOptions {

        /// Max runs for Gauss Newton
        int max_runs = 20;

        /// Init lambda for LM optimization
        double init_lamda = 1e-3;

        /// Max lambda for LM optimization
        double max_lamda = 1e10;

        /// Cutoff for dx increment to consider as converged
        double min_dx = 1e-6;

        /// Cutoff for cost decrement to consider as converged
        double min_dcost = 1e-6;

        /// Multiplier to increase/decrease lambda
        double lam_mult = 10;

        /// Minimum distance to accept triangulated features
        double min_dist = 0.25;

        /// Minimum distance to accept triangulated features
        double max_dist = 40;

        /// Max baseline ratio to accept triangulated features
        double max_baseline = 40;

        /// Max condition number of linear triangulation matrix accept triangulated features
        double max_cond_number = 1000;

        /// Nice print function of what parameters we have loaded
        void print() {
            printf("\t- max_runs: %d\n", max_runs);
            printf("\t- init_lamda: %.3f\n", init_lamda);
            printf("\t- max_lamda: %.3f\n", max_lamda);
            printf("\t- min_dx: %.7f\n", min_dx);
            printf("\t- min_dcost: %.7f\n", min_dcost);
            printf("\t- lam_mult: %.3f\n", lam_mult);
            printf("\t- min_dist: %.3f\n", min_dist);
            printf("\t- max_dist: %.3f\n", max_dist);
            printf("\t- max_baseline: %.3f\n", max_baseline);
            printf("\t- max_cond_number: %.3f\n", max_cond_number);
        }

    };

}

#endif // INITIALIZEROPTIONS_H
