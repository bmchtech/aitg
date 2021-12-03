const app = new Moon({
    el: "#moon-app",
    data: {
        query: "test",
        result_query: "",
        results: [],
        selected_result: -1,
    },
    methods: {
        submit_search: function(a, b) {
            console.log("submit_search", a, b);
            this.set("results", []);
            this.set("selected_result", -1);
            // do search
            let query = this.get("query");
            console.log("searching for: " + query);
            this.set("result_query", query);
            this.set("results", [
                {
                    title: "doc1",
                    match: "context free grammars are <b>a type of context-free</b> language. lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
                    score: 0.33,
                }
            ]);
        }
    },
});

// Moon.component("search-ui", {
//     template: `
//     `,
//     data: {
//         query: "",
//         results: [],
//     }
// });