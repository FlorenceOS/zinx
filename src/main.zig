const std = @import("std");

var source_files = std.ArrayListUnmanaged(SourceFile){};
var tokens = std.ArrayListUnmanaged(Token){};
var expressions = std.ArrayListUnmanaged(Expression){};

var gpa = std.heap.GeneralPurposeAllocator(.{}){
    .backing_allocator = std.heap.page_allocator
};
const alloc = gpa.allocator();

const NO_SCOPE = 0xFFFFFFFF;

var host_path: ?[]const u8 = null;
var host_string_expr: ?u32 = null;

const SourceFile = struct {
    buffer: [:0]const u8,
    realpath: [*:0]const u8,
    dir: std.fs.Dir,
    top_level_value: u32,
    tokens: ?SourceBound,
};

fn open_file(dir: std.fs.Dir, path: [:0]const u8, bound: ?SourceBound) !u32 {
    var realpath_buf: [std.os.PATH_MAX]u8 = undefined;
    const rp = dir.realpathZ(path, &realpath_buf) catch |err| {
        if(err == error.FileNotFound) {
            report_error(bound, "File not found: {s}", .{path});
        }
        return err;
    };

    for(source_files.items, 0..) |sf, idx| {
        if(std.mem.eql(u8, std.mem.span(sf.realpath), rp)) {
            return @intCast(u32, idx);
        }
    }

    const fd = try dir.openFileZ(path, .{});
    const file_size = try fd.getEndPos();
    const mem = try std.os.mmap(
        null,
        (file_size + 0x1000) & ~@as(usize, 0xFFF),
        std.os.PROT.READ,
        std.os.MAP.SHARED,
        fd.handle,
        0,
    );

    const new_dir = try std.fs.openDirAbsolute(std.fs.path.dirname(rp).?, .{});

    try source_files.append(alloc, .{
        .buffer = @ptrCast([:0]const u8, mem),
        .realpath = try alloc.dupeZ(u8, rp),
        .dir = new_dir,
        .top_level_value = NO_SCOPE,
        .tokens = null,
    });

    return @intCast(u32, source_files.items.len - 1);
}

fn add_expr(expr: ExpressionValue) u32 {
    expressions.append(alloc, .{.value = expr}) catch unreachable;
    return @intCast(u32, expressions.items.len - 1);
}

const SourceLocation = struct {
    source_file: u32,
    line: u32,
    column: u32,
    offset: u32,

    fn prev(self: @This()) ?SourceLocation {
        if(self.offset == 0) return null;
        const up_line = source_files.items[self.source_file].buffer[self.offset - 1] == '\n';
        return .{
            .source_file = self.source_file,
            .line = if(up_line) self.line - 1 else self.line,
            .column = if(up_line) undefined else self.column - 1,
            .offset = self.offset - 1,
        };
    }
};

const SourceBound = struct {
    begin: u32,
    end: u32,

    fn combine(self: @This(), other: @This()) @This() {
        std.debug.assert(tokens.items[self.begin].loc.source_file == tokens.items[self.begin].loc.source_file);
        if(self.begin < other.begin) {
            return .{
                .begin = self.begin,
                .end = other.end,
            };
        } else {
            return .{
                .begin = other.begin,
                .end = self.end,
            };
        }
    }

    fn consume_token(self: *@This()) ?u32 {
        if(self.begin > self.end) {
            return null;
        }
        self.begin += 1;
        return self.begin - 1;
    }

    fn peek_token(self: *@This()) ?Token {
        if(self.begin > self.end) return null;
        return tokens.items[self.begin];
    }

    fn extend_right(self: @This(), num: u32) @This() {
        return .{
            .begin = self.begin,
            .end = self.end + num,
        };
    }
};

const Token = struct {
    const Value = enum(u32) {
        lparen,
        rparen,
        lcurly,
        rcurly,
        lsquare,
        rsquare,
        identifier,
        dot,
        eql,
        comma,
        plain_string_start,
        plain_string_chunk,
        plain_string_end,
        multi_string_start,
        multi_string_chunk,
        multi_string_end,
        bad_char,
    };

    loc: SourceLocation,
    value: Value,

    fn length(self: @This()) u32 {
        return switch(self.value) {
            .bad_char,
            .lparen, .rparen, .lcurly, .rcurly, .lsquare, .rsquare,
            .dot, .eql, .comma, .plain_string_start, .multi_string_start,
            .plain_string_end,
            => 1,
            .multi_string_end => 0,
            else => {
                var stream = Stream{
                    .buffer = source_files.items[self.loc.source_file].buffer,
                    .pos = self.loc,
                };
                stream.states.appendAssumeCapacity(switch(self.value) {
                    .plain_string_chunk => .{.plain_string = {}},
                    .multi_string_chunk => .{.multi_string = {}},
                    else => .{.normal = 0},
                });
                _ = stream.generate_token() catch unreachable;
                return stream.pos.offset - self.loc.offset;
            }
        };
    }

    fn identifier_string(self: @This()) ![]const u8 {
        if(self.value != .identifier) return error.NotString;
        const sf = source_files.items[self.loc.source_file];
        return sf.buffer[self.loc.offset..][0..self.length()];
    }

    fn append_string_value(self: @This(), str: *std.ArrayListUnmanaged(u8)) void {
        var offset: usize = self.loc.offset;
        var end = offset + self.length();
        const buf = source_files.items[self.loc.source_file].buffer;

        switch(self.value) {
            .plain_string_chunk => {
                while(offset < end) {
                    switch(buf[offset]) {
                        '\\' => {
                            offset += 1;
                        },
                        '$' => {
                            if(buf[offset + 1] == '{') break;
                        },
                        else => {},
                    }
                    str.append(alloc, buf[offset]) catch unreachable;
                    offset += 1;
                }
            },
            .multi_string_chunk => {
                var active = true;
                while(offset < end) {
                    if(active) {
                        if(buf[offset] == '$' and buf[offset + 1] == '{') {
                            break;
                        }
                        str.append(alloc, buf[offset]) catch unreachable;
                        switch(buf[offset]) {
                            '\n' => active = false,
                            else => {},
                        }
                        offset += 1;
                    } else {
                        offset = std.mem.indexOfPosLinear(u8, buf[0..end], offset, "\\\\") orelse return;
                        active = true;
                    }
                }
            },
            else => unreachable,
        }
    }
};

fn token_bound(idx: u32) SourceBound {
    return .{
        .begin = idx,
        .end = idx,
    };
}

fn message_impl(
    bound: ?SourceBound,
    comptime message: []const u8,
    fmtargs: anytype,
) void {
    std.debug.print(message ++ " in ", fmtargs);
    if(bound) |b| {
        const btok = tokens.items[b.begin];
        const sf = source_files.items[btok.loc.source_file];
        std.debug.print("{s}:{d}:{d}:\n", .{
            sf.realpath,
            btok.loc.line,
            btok.loc.column,
        });
        const line_start = blk: {
            const idx = std.mem.lastIndexOfScalar(u8, sf.buffer[0..btok.loc.offset], '\n') orelse break :blk 0;
            break :blk idx + 1;
        };
        const line_end = std.mem.indexOfScalarPos(u8, sf.buffer, btok.loc.offset, '\n') orelse sf.buffer.len;
        const tok_len = btok.length();
        std.debug.print("{s}\x1b[31;1m{s}\x1b[0m{s}\n", .{
            sf.buffer[line_start..btok.loc.offset],
            sf.buffer[btok.loc.offset..][0..tok_len],
            sf.buffer[btok.loc.offset + tok_len..line_end],
        });
    } else {
        std.debug.print("???\n", .{});
    }
}

fn report_error(bound: ?SourceBound, comptime message: []const u8, fmtargs: anytype) void {
    message_impl(bound, "\x1b[31;1merror\x1b[0m: " ++ message, fmtargs);
}

fn note(bound: ?SourceBound, comptime message: []const u8, fmtargs: anytype) void {
    message_impl(bound, "\x1b[37;2;1mnote\x1b[0m: " ++ message, fmtargs);
}

const Stream = struct {
    const State = union(enum) {
        normal: usize,
        plain_string,
        multi_string,
    };

    buffer: [*:0]const u8,
    states: std.BoundedArray(State, 1024) = .{},
    pos: SourceLocation,

    fn current_state(self: *@This()) *State {
        return &self.states.slice()[self.states.slice().len - 1];
    }

    fn peek(self: *@This()) u8 {
        if(self.buffer[self.pos.offset] == 0) return 0;
        return self.buffer[self.pos.offset];
    }

    fn consume(self: *@This()) u8 {
        const result = self.peek();
        if(result != 0) {
            if(result == '\n') {
                self.pos.line += 1;
                self.pos.column = 1;
            } else {
                self.pos.column += 1;
            }
            self.pos.offset += 1;
        }
        return result;
    }

    fn skip_whitespace(self: *@This()) void {
        while(true) {
            switch(self.peek()) {
                ' ', '\t', '\r', '\n' => {
                    _ = self.consume();
                },
                else => return,
            }
        }
    }

    fn bad_consumed(self: *@This(), c: u8) !noreturn {
        try tokens.append(alloc, .{
            .value = .bad_char,
            .loc = self.pos.prev().?,
        });
        const faketok = @intCast(u32, tokens.items.len - 1);
        report_error(
            .{.begin = faketok, .end = faketok},
            "Unexpected character: {c} (0x{X}){s}",
            .{c, c, if(c == 0) " (end of file)" else ""},
        );
        return error.BadCharacter;
    }

    fn generate_token(self: *@This()) !?Token {
        var start_pos = self.pos;
        const value: Token.Value = value: { switch(self.current_state().*) {
            .plain_string => {
                var length: usize = 0;
                while(true) {
                    switch(self.peek()) {
                        0 => {
                            _ = self.consume();
                            try self.bad_consumed(0);
                        },
                        '\\' => {
                            @panic("TODO: String escapes");
                        },
                        '$' => {
                            _ = self.consume();
                            if(self.peek() == '{') {
                                _ = self.consume();
                                try self.states.append(.{.normal = 0});
                                if(length == 0) {
                                    return @call(.always_tail, generate_token, .{self});
                                } else {
                                    break :value .plain_string_chunk;
                                }
                            } else {
                                length += 1;
                            }
                        },
                        '"' => {
                            if(length == 0) {
                                _ = self.consume();
                                _ = self.states.pop();
                                break :value .plain_string_end;
                            } else {
                                break :value .plain_string_chunk;
                            }
                        },
                        else => {
                            _ = self.consume();
                            length += 1;
                        },
                    }
                }
            },
            .multi_string => {
                var active = true;
                while(true) {
                    if(active) {
                        switch(self.consume()) {
                            0 => {
                                self.states.len = 1;
                                break :value .multi_string_chunk;
                            },
                            '\\' => {
                                @panic("TODO: String escapes");
                            },
                            '\n' => active = false,
                            '$' => {
                                if(self.peek() == '{') {
                                    _ = self.consume();
                                    try self.states.append(.{.normal = 0});
                                    break :value .multi_string_chunk;
                                }
                            },
                            else => { },
                        }
                    } else {
                        const potential_end = self.pos;
                        self.skip_whitespace();
                        if(self.peek() == '\\') {
                            _ = self.consume();
                            active = true;
                            continue;
                        } else {
                            self.pos = potential_end;
                            _ = self.states.pop();
                            break :value .multi_string_end;
                        }
                    }
                }
            },
            .normal => |*depth| {
                self.skip_whitespace();
                start_pos = self.pos;
                switch(self.consume()) {
                    0 => return null,
                    '.' => break :value .dot,
                    '=' => break :value .eql,
                    ',' => break :value .comma,
                    '(' => break :value .lparen,
                    ')' => break :value .rparen,
                    '[' => break :value .lsquare,
                    ']' => break :value .rsquare,
                    '{' => {
                        depth.* += 1;
                        break :value .lcurly;
                    },
                    '}' => {
                        if(depth.* == 0) {
                            _ = self.states.pop();
                            return @call(.always_tail, generate_token, .{self});
                        } else {
                            depth.* -= 1;
                            break :value .rcurly;
                        }
                    },
                    '\\' => {
                        try self.states.append(.{.multi_string = {}});
                        break :value .multi_string_start;
                    },
                    '"' => {
                        try self.states.append(.{.plain_string = {}});
                        break :value .plain_string_start;
                    },
                    'a'...'z',
                    'A'...'Z',
                    '@', '_',
                    => {
                        while(true) {
                            switch(self.peek()) {
                                'a'...'z',
                                'A'...'Z',
                                '@', '_',
                                '0'...'9',
                                => _ = self.consume(),
                                else => break :value .identifier,
                            }
                        }
                    },
                    else => |o| {
                        try self.bad_consumed(o);
                        unreachable;
                    }, // wtf comma here
                }
            },
        }};

        return .{
            .loc = start_pos,
            .value = value,
        };
    }
};

fn tokenize_file(source_file: u32) !SourceBound {
    const begin = @intCast(u32, tokens.items.len);

    var stream = Stream {
        .buffer = source_files.items[source_file].buffer,
        .pos = .{
            .source_file = source_file,
            .line = 1,
            .column = 1,
            .offset = 0,
        },
    };
    stream.states.appendAssumeCapacity(.{.normal = 0});

    while(try stream.generate_token()) |tok| {
        tokens.append(alloc, tok) catch unreachable;
    }

    return .{
        .begin = begin,
        .end = @intCast(u32, tokens.items.len - 1),
    };
}

fn parse_dict(tok: *SourceBound, scope: u32) !u32 {
    const new_scope = add_expr(.{.dict = .{.parent = scope}});
    while(true) {
        const ident = tok.consume_token() orelse return new_scope;
        switch(tokens.items[ident].value) {
            .rcurly => return new_scope,
            .identifier => {
                const s = tokens.items[ident].identifier_string() catch unreachable;
                const ass = tok.consume_token().?;
                if(tokens.items[ass].value != .eql) {
                    report_error(token_bound(ass), "Expected '=' after identifier", .{});
                    return error.BadToken;
                } else {
                    const expr = try parse_expr(tok, new_scope);
                    try expressions.items[new_scope].value.dict.values.putNoClobber(alloc, s, expr);
                }
            },
            else => |o| {
                report_error(token_bound(ident), "Expected identifier in dict (got tag {s})", .{@tagName(o)});
                return error.BadToken;
            },
        }
    }
}

fn parse_expr(tok: *SourceBound, scope: u32) anyerror!u32 {
    const primary_expr = tok.consume_token() orelse {
        report_error(
            null,
            "Unexpected end of file",
            .{},
        );
        return error.UnexpectedEOF;
    };
    var lhs = switch(tokens.items[primary_expr].value) {
        .lparen => { // Function
            @panic("TODO: Function");
        },
        .lcurly => try parse_dict(tok, scope),
        .lsquare => { // List
            @panic("TODO: List");
        },
        .plain_string_start, .multi_string_start => blk: {
            var string_parts: std.ArrayListUnmanaged(ExpressionValue.StringPart) = .{};
            while(true) {
                switch(tok.peek_token().?.value) {
                    .plain_string_end, .multi_string_end => {
                        _ = tok.consume_token();
                        break :blk add_expr(.{.string_parts = try string_parts.toOwnedSlice(alloc)});
                    },
                    .plain_string_chunk, .multi_string_chunk => {
                        try string_parts.append(alloc, .{.string_chunk = tok.consume_token().?});
                    },
                    else => {
                        try string_parts.append(alloc, .{.expression_chunk = try parse_expr(tok, scope)});
                    },
                }
            }
        },
        .identifier => add_expr(.{.identifier = primary_expr}),
        .plain_string_chunk, .plain_string_end,
        .multi_string_chunk, .multi_string_end,
        .rparen, .rsquare, .rcurly, .bad_char, .dot, .eql, .comma,
        => |o| {
            report_error(
                .{.begin = primary_expr, .end = primary_expr},
                "Unexpected expression token (tag {s})",
                .{@tagName(o)},
            );
            return error.BadToken;
        },
    };
    while (true) {
        switch((tok.peek_token() orelse return lhs).value) {
            .dot => {
                _ = tok.consume_token();
                const ident = tok.consume_token().?;
                if(tokens.items[ident].value != .identifier) {
                    report_error(
                        token_bound(ident),
                        "Expected identifier after '.' expression",
                        .{},
                    );
                    return error.BadToken;
                }
                const new = add_expr(.{.member_access = .{
                    .src = lhs,
                    .ident = ident,
                }});
                lhs = new;
            },
            .lparen => @panic("TODO: Function call"),
            .lsquare => @panic("TODO: Subscript"),

            .identifier,
            .rparen, .lcurly, .rcurly, .rsquare, .eql, .comma,
            .plain_string_start, .plain_string_chunk, .plain_string_end,
            .multi_string_start, .multi_string_chunk, .multi_string_end,
            .bad_char,
            => {
                return lhs;
            },
        }
    }
}

const Scope = struct {
    values: std.StringHashMapUnmanaged(u32) = .{}, // exprs
    parent: u32,

    fn lookup(self: @This(), name: []const u8) ?u32 {
        if(self.values.get(name)) |v| {
            return v;
        } else {
            if(self.parent == NO_SCOPE) {
                return null;
            } else {
                return @call(.always_tail, expressions.items[self.parent].value.dict.lookup, .{name});
            }
        }
    }
};

const ExpressionValue = union(enum) {
    const StringPart = union(enum) {
        string_chunk: u32, // token
        expression_chunk: u32, // expr
    };

    // Needs no resolution
    invalid,
    function: struct {
        // All map values must have an invalid expression
        // Can be cloned when doing a function call with the correct values
        argument_template: Scope, // exprs
        body: u32, // expr
    },
    string: struct {
        orig_bound: SourceBound,
        value: []const u8,
    },
    alias: struct {
        orig_bound: SourceBound,
        expr_value: u32,
    },
    host: SourceBound,
    host_string_underlying,

    // Lazily resolves children upon member access
    dict: Scope,

    // Needs to resolve child expressions
    list: []const u32,
    string_parts: []const StringPart,

    // Unresolved values, has to be a value above after resolution
    call: struct {
        callee: u32, // expr
        args: std.StringHashMapUnmanaged(u32),
    },
    member_access: struct {
        src: u32, // expr
        ident: u32, // token
    },
    subscript: struct {
        src: u32, // expr
        idx: u32, // expr
    },
    identifier: u32, // token
    override: struct {
        lhs: u32, // expr
        rhs: u32, // expr
    },
    source_file: struct {
        bound: SourceBound,
        file: u32, // source file
    },
};

const Expression = struct {
    value: ExpressionValue,
    resolving: bool = false,
    resolved: bool = false,

    fn make_alias(self: *@This(), dict: u32, value: u32) !void {
        try expressions.items[value].resolve(dict);
        const b = self.bound();
        self.value = .{.alias = .{
            .orig_bound = b,
            .expr_value = switch(expressions.items[value].value) {
                .alias => |a| a.expr_value,
                else => value,
            }},
        };
    }

    fn dealias(self: @This()) @This() {
        switch(self.value) {
            .alias => |a| return expressions.items[a.expr_value],
            else => return self,
        }
    }

    fn lookup(self: @This(), key: []const u8) !?u32 {
        switch(self.value) {
            .host => {
                return host_string_expr.?;
            },
            .dict => |d| {
                if(d.values.get(key)) |val| {
                    return val;
                } else {
                    return error.BadKey;
                }
            },
            else => |o| {
                report_error(
                    self.bound(),
                    "Expected dict, found {s}",
                    .{@tagName(o)},
                );
                return error.KeyNotFound;
            },
        }
    }

    fn to_string(self: *@This()) ![]const u8 {
        var result = std.ArrayListUnmanaged(u8){};
        try self.append_string_value(&result);
        return result.toOwnedSlice(alloc);
    }

    fn resolve(self: *@This(), scope: u32) anyerror!void {
        if(self.resolved) return;
        if(self.resolving) {
            report_error(
                self.bound(),
                "Infinite recursion on evaluating expression",
                .{},
            );
            return error.CircularDependency;
        }
        self.resolving = true;
        defer self.resolving = false;
        defer self.resolved = true;
        errdefer {
            note(self.bound(), "While evaluating expression", .{});
        }

        switch(self.value) {
            .invalid => unreachable,
            .dict, .function, .string, .alias, .host, .host_string_underlying,
            => {},
            .identifier => |i| {
                std.debug.assert(scope != NO_SCOPE);
                const is = tokens.items[i].identifier_string() catch unreachable;
                return try self.make_alias(scope, expressions.items[scope].dealias().value.dict.lookup(is) orelse {
                    report_error(token_bound(i), "Identifier '{s}' not found!", .{is});
                    return error.BadIdentifier;
                });
            },
            .override => {
                @panic("TODO: Override");
            },
            .source_file => |sf| {
                if(source_files.items[sf.file].tokens == null) {
                    source_files.items[sf.file].tokens = try tokenize_file(sf.file);
                }
                if(source_files.items[sf.file].top_level_value == NO_SCOPE) {
                    var toks = source_files.items[sf.file].tokens.?;
                    // while(toks.consume_token()) |tok| {
                    //     std.debug.print("Token tag: {s}\n", .{@tagName(tokens.items[tok].value)});
                    // }
                    source_files.items[sf.file].top_level_value = try parse_dict(&toks, NO_SCOPE);
                }
                try self.make_alias(NO_SCOPE, source_files.items[sf.file].top_level_value);
            },
            .list => |l| {
                for(l) |expr| {
                    try expressions.items[expr].resolve(scope);
                }
            },
            .call => |_| {
                @panic("TODO: Call");
            },
            .member_access => |ma| {
                try expressions.items[ma.src].resolve(scope);
                const str = tokens.items[ma.ident].identifier_string() catch unreachable;
                try self.make_alias(ma.src, try expressions.items[ma.src].dealias().lookup(str) orelse {
                    report_error(
                        token_bound(ma.ident),
                        "Key '{s}' not found in dict",
                        .{str},
                    );
                    return error.BadKey;
                });
            },
            .string_parts => |sp| {
                for(sp) |part| {
                    switch(part) {
                        .expression_chunk => |e|{
                            try expressions.items[e].resolve(scope);
                        },
                        else => {},
                    }
                }
            },
            .subscript => |ss| {
                try expressions.items[ss.idx].resolve(scope);
                var str = std.ArrayListUnmanaged(u8){};
                defer str.deinit(alloc);
                try expressions.items[ss.idx].dealias().append_string_value(&str);

                try expressions.items[ss.src].resolve(scope);
                try self.make_alias(ss.src, try expressions.items[ss.src].dealias().lookup(str.items) orelse {
                    report_error(
                        expressions.items[ss.idx].bound(),
                        "Key '{s}' not found in dict",
                        .{str.items},
                    );
                    return error.BadKey;
                });
            },
        }
    }

    fn append_string_value(self: @This(), str: *std.ArrayListUnmanaged(u8)) !void {
        switch(self.dealias().value) {
            .string => |s| {
                try str.appendSlice(alloc, s.value);
            },
            .string_parts => |sp| {
                for(sp) |item| {
                    switch(item) {
                        .string_chunk => |sc| {
                            tokens.items[sc].append_string_value(str);
                        },
                        .expression_chunk => |ec| {
                            expressions.items[ec].append_string_value(str) catch unreachable;
                        },
                    }
                }
            },
            else => |o| {
                report_error(
                    self.bound(),
                    "Expexted string, found {s}",
                    .{@tagName(o)},
                );
                return error.BadValue;
            },
        }
    }

    fn bound(self: @This()) SourceBound {
        return switch(self.value) {
            .invalid, .host_string_underlying,
            .function, .dict, .list, .string_parts, .call,
            => unreachable,

            .member_access => |ma| return expressions.items[ma.src].bound().combine(token_bound(ma.ident)),

            .override => |o| expressions.items[o.lhs].bound().combine(expressions.items[o.rhs].bound()),

            .subscript => |ss| expressions.items[ss.src].bound().combine(
                expressions.items[ss.idx].bound().extend_right(1),
            ),

            .source_file => |sf| sf.bound,

            inline .string, .alias => |v| v.orig_bound,
            .host => |b| b,
            .identifier,
            => |t| token_bound(t),
        };
    }
};

pub fn main() !void {
    try tokens.ensureTotalCapacity(alloc, 8 * 1024 * 1024);
    try expressions.ensureTotalCapacity(alloc, 8 * 1024 * 1024);
    try source_files.ensureTotalCapacity(alloc, 1024 * 1024);

    var it = std.process.args();

    var positional_args = std.BoundedArray([:0]const u8, 3){};

    while(it.next()) |arg| {
        if(std.mem.startsWith(u8, arg, "--host-dir=")) {
            host_path = arg[11..];
        } else if(std.mem.startsWith(u8, arg, "--build-dir=")) {
            std.debug.print("Build dir is {s}\n", .{arg[12..]});
        } else {
            try positional_args.append(arg);
        }
    }

    host_string_expr = add_expr(.{.host_string_underlying = {}});

    const root_file = try open_file(std.fs.cwd(), positional_args.slice()[1], null);
    const root_expr = add_expr(.{.source_file = .{
        .file = root_file,
        .bound = .{
            .begin = 0,
            .end = 0,
        },
    }});
    try expressions.items[root_expr].resolve(NO_SCOPE);

    // Ugly hack to parse cmdline as expression
    try source_files.append(alloc, .{
        .buffer = positional_args.slice()[2],
        .realpath = "<cmdline>",
        .dir = std.fs.cwd(),
        .top_level_value = NO_SCOPE,
        .tokens = null,
    });
    var cmdline_tokens = try tokenize_file(@intCast(u32, source_files.items.len - 1));
    const cmdline_expr = try parse_expr(&cmdline_tokens, root_expr);
    try expressions.items[cmdline_expr].resolve(root_expr);
    const result = try expressions.items[cmdline_expr].to_string();
    std.debug.print("{s}\n",.{result});
}
