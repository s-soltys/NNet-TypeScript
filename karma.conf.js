module.exports = function (config) {
    let browsers = ["Chrome"];

    if (process.env.TRAVIS) {
        browsers = ['Chrome_travis_ci'];
    }

    config.set({
        frameworks: ["jasmine", "karma-typescript"],
        files: [
            { pattern: "src/**/*.ts" },
            { pattern: "tests/**/*.ts" },
        ],
        preprocessors: {
            "**/*.ts": ["karma-typescript"],
        },
        reporters: ["progress", "karma-typescript"],
        browsers: browsers,
        singleRun: true,
        customLaunchers: {
            Chrome_travis_ci: {
                base: 'Chrome',
                flags: ['--no-sandbox']
            }
        },
    });
};