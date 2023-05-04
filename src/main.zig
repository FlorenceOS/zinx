const std = @import("std");

var source_files = std.SegmentedList(SourceFile, 1024){};
var tokens = std.SegmentedList(Token, 32 * 1024){};
var expressions = std.SegmentedList(Expression, 32 * 1024){};

var gpa = std.heap.GeneralPurposeAllocator(.{}){
    .backing_allocator = std.heap.page_allocator
};
const alloc = gpa.allocator();

const NO_SCOPE = 0xFFFFFFFF;
const NO_ARG_VALUE = 0xFFFFFFFF;

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

    {
        var result: u32 = 0;
        var it = source_files.iterator(0);
        while(it.next()) |sf| {
            if(std.mem.eql(u8, std.mem.span(sf.realpath), rp)) {
                return result;
            }
            result += 1;
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

    return @intCast(u32, source_files.count() - 1);
}

fn add_expr(expr: ExpressionValue) u32 {
    expressions.append(alloc, .{.value = expr}) catch unreachable;
    return @intCast(u32, expressions.count() - 1);
}

const SourceLocation = struct {
    source_file: u32,
    line: u32,
    column: u32,
    offset: u32,

    fn prev(self: @This()) ?SourceLocation {
        if(self.offset == 0) return null;
        const up_line = source_files.at(self.source_file).buffer[self.offset - 1] == '\n';
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
        std.debug.assert(tokens.at(self.begin).loc.source_file == tokens.at(other.begin).loc.source_file);
        return .{
            .begin = std.math.min(self.begin, other.begin),
            .end = std.math.max(self.end, other.end),
        };
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
        return tokens.at(self.begin).*;
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
                    .buffer = source_files.at(self.loc.source_file).buffer,
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
        const sf = source_files.at(self.loc.source_file);
        return sf.buffer[self.loc.offset..][0..self.length()];
    }

    fn append_string_value(self: @This(), str: *std.ArrayListUnmanaged(u8)) void {
        var offset: usize = self.loc.offset;
        var end = offset + self.length();
        const buf = source_files.at(self.loc.source_file).buffer;

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
        const btok = tokens.at(b.begin);
        const sf = source_files.at(btok.loc.source_file);
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
        multi_string_end,
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
                '/' => {
                    while(true) {
                        switch(self.consume()) {
                            '\n' => break,
                            else => { },
                        }
                    }
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
        const faketok = @intCast(u32, tokens.count() - 1);
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
                var length: usize = 0;
                while(true) {
                    switch(self.consume()) {
                        0 => {
                            if(length == 0) {
                                _ = self.states.pop();
                                break :value .multi_string_end;
                            } else {
                                break :value .multi_string_chunk;
                            }
                        },
                        '\\' => {
                            @panic("TODO: String escapes");
                        },
                        '\n' => {
                            length += 1;
                            const potential_end = self.pos;
                            self.skip_whitespace();
                            if(self.peek() == '\\') {
                                _ = self.consume();
                            } else {
                                self.pos = potential_end;
                                self.current_state().* = .multi_string_end;
                                break :value .multi_string_chunk;
                            }
                        },
                        '$' => {
                            if(self.peek() == '{') {
                                _ = self.consume();
                                try self.states.append(.{.normal = 0});
                                if(length == 0) {
                                    return @call(.always_tail, generate_token, .{self});
                                } else {
                                    break :value .multi_string_chunk;
                                }
                            } else {
                                length += 1;
                            }
                        },
                        else => {
                            length += 1;
                        },
                    }
                }
            },
            .multi_string_end => {
                _ = self.states.pop();
                break :value .multi_string_end;
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
    const begin = @intCast(u32, tokens.count());

    var stream = Stream {
        .buffer = source_files.at(source_file).buffer,
        .pos = .{
            .source_file = source_file,
            .line = 1,
            .column = 1,
            .offset = 0,
        },
    };
    stream.states.appendAssumeCapacity(.{.normal = 0});

    while(try stream.generate_token()) |tok| {
        //std.debug.print("Token: {any}\n", .{tok});
        tokens.append(alloc, tok) catch unreachable;
    }

    return .{
        .begin = begin,
        .end = @intCast(u32, tokens.count() - 1),
    };
}

fn parse_dict(tok: *SourceBound) !u32 {
    const new_scope = add_expr(.{.dict = .{.parent = undefined}});
    while(true) {
        const ident = tok.consume_token() orelse return new_scope;
        switch(tokens.at(ident).value) {
            .rcurly => return new_scope,
            .identifier => {
                const s = tokens.at(ident).identifier_string() catch unreachable;
                const ass = tok.consume_token().?;
                if(tokens.at(ass).value != .eql) {
                    report_error(token_bound(ass), "Expected '=' after identifier", .{});
                    return error.BadToken;
                } else {
                    const expr = try parse_expr(tok);
                    try expressions.at(new_scope).value.dict.values.putNoClobber(alloc, s, expr);
                }
            },
            else => |o| {
                report_error(token_bound(ident), "Expected identifier in dict (got tag {s})", .{@tagName(o)});
                return error.BadToken;
            },
        }
    }
}

fn parse_expr(tok: *SourceBound) anyerror!u32 {
    const primary_expr = tok.consume_token() orelse {
        report_error(
            null,
            "Unexpected end of file",
            .{},
        );
        return error.UnexpectedEOF;
    };
    var lhs = switch(tokens.at(primary_expr).value) {
        .lparen => blk: { // Function
            var argument_template = std.StringHashMapUnmanaged(u32){};
            while(tok.peek_token().?.value != .rparen) {
                const ident = tok.consume_token().?;
                if(tokens.at(ident).value != .identifier) {
                    report_error(
                        token_bound(ident),
                        "Expected argument name identifier",
                        .{},
                    );
                    return error.BadToken;
                }
                try argument_template.putNoClobber(
                    alloc,
                    tokens.at(ident).identifier_string() catch unreachable,
                    NO_ARG_VALUE
                );
                if(tok.peek_token().?.value == .comma) {
                    _ = tok.consume_token();
                } else {
                    break;
                }
            }
            std.debug.assert(tokens.at(tok.consume_token().?).value == .rparen);
            const f = add_expr(.{.function = .{
                .argument_template = .{
                    .values = argument_template,
                    .parent = undefined,
                },
                .body = undefined,
            }});
            expressions.at(f).value.function.body = try parse_expr(tok);
            break :blk f;
        },
        .lcurly => try parse_dict(tok),
        .lsquare => blk: { // List
            var result = std.ArrayListUnmanaged(u32){};
            var lsquare_bound = tok.*;
            while(tok.peek_token().?.value != .rsquare) {
                const expr = try parse_expr(tok);
                try result.append(alloc, expr);
                if(tok.peek_token().?.value == .comma) {
                    _ = tok.consume_token();
                } else {
                    break;
                }
            }
            std.debug.assert(tokens.at(tok.consume_token().?).value == .rsquare);
            break :blk add_expr(.{.list = .{
                .lsquare_bound = lsquare_bound,
                .items = try result.toOwnedSlice(alloc),
            }});
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
                        try string_parts.append(alloc, .{.expression_chunk = try parse_expr(tok)});
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
                if(tokens.at(ident).value != .identifier) {
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
            .lparen => {
                var result = std.StringHashMapUnmanaged(u32){};
                _ = tok.consume_token();
                while(tok.peek_token().?.value != .rparen) {
                    const ident = tok.consume_token().?;
                    if(tokens.at(ident).value != .identifier) {
                        report_error(
                            token_bound(ident),
                            "Expected argument name identifier",
                            .{},
                        );
                        return error.BadToken;
                    }
                    const ass = tok.consume_token().?;
                    if(tokens.at(ass).value != .eql) {
                        report_error(
                            token_bound(ass),
                            "Expected '=' after argument name",
                            .{},
                        );
                        return error.BadToken;
                    }
                    const expr = try parse_expr(tok);
                    try result.putNoClobber(
                        alloc,
                        tokens.at(ident).identifier_string() catch unreachable,
                        expr
                    );
                    if(tok.peek_token().?.value == .comma) {
                        _ = tok.consume_token();
                    } else {
                        break;
                    }
                }
                std.debug.assert(tokens.at(tok.consume_token().?).value == .rparen);
                lhs = add_expr(.{.call = .{
                    .callee = lhs,
                    .args = result,
                }});
            },
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
                var parent = expressions.at(self.parent);
                if(parent.value == .alias) {
                    parent = expressions.at(parent.value.alias.expr_value);
                }

                switch (parent.value) {
                    .dict => |scope| return @call(.always_tail, scope.lookup, .{name}),
                    .function => |func| return @call(.always_tail, func.argument_template.lookup, .{name}),
                    inline else => |_, tag| @panic("Invalid scope parent: " ++ @tagName(tag)),
                }
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
    builtin_function: struct {
        token: u32,
        value: enum {
            transform,
            join,
        },
    },
    host_string_underlying,

    // Lazily resolves children upon member access
    dict: Scope,

    // Needs to resolve child expressions
    list: struct {
        lsquare_bound: SourceBound,
        items: []const u32,
    },

    string_parts: []const StringPart,
    transformed_list: struct {
        orig_bound: SourceBound,
        elements: u32,
        transform: u32,
    },
    joined_string: struct {
        orig_bound: SourceBound,
        elements: u32,
        separator: u32,
        scope: u32,
    },

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
        try expressions.at(value).resolve(dict);
        const b = self.bound();
        self.value = .{.alias = .{
            .orig_bound = b,
            .expr_value = switch(expressions.at(value).value) {
                .alias => |a| a.expr_value,
                else => value,
            }},
        };
    }

    fn dealias(self: @This()) @This() {
        switch(self.value) {
            .alias => |a| return expressions.at(a.expr_value).*,
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

    fn deep_copy_value(value: u32) !ExpressionValue {
        switch(expressions.at(value).value) {
            .dict => |d| return .{.dict = .{
                .values = try d.values.clone(alloc),
                .parent = d.parent,
            }},
            .function => |f| return .{.function = .{
                // We don't need to deep copy the argument dicts because they are deep
                // copied before they are modified, during the call anyways
                .argument_template = f.argument_template,
                .body = add_expr(try deep_copy_value(f.body)),
            }},
            .list => |l| {
                const result = try alloc.dupe(u32, l.items);
                for(result) |*v| {
                    v.* = add_expr(try deep_copy_value(v.*));
                }
                return .{.list = .{.lsquare_bound = l.lsquare_bound, .items = result}};
            },
            .string_parts => |sp| {
                const result = try alloc.dupe(ExpressionValue.StringPart, sp);
                for(result) |*v| {
                    switch(v.*) {
                        .expression_chunk => |*ec| ec.* = add_expr(try deep_copy_value(ec.*)),
                        else => { },
                    }
                }
                return .{.string_parts = result};
            },
            .transformed_list => unreachable,
            .joined_string => unreachable,
            .call => |c| {
                var result = c;
                result.callee = add_expr(try deep_copy_value(result.callee));
                result.args = try result.args.clone(alloc);
                var it = result.args.iterator();
                while(it.next()) |a| {
                    a.value_ptr.* = add_expr(try deep_copy_value(a.value_ptr.*));
                }
                return .{.call = result};
            },
            .member_access => |ma| return .{.member_access = .{
                .src = add_expr(try deep_copy_value(ma.src)),
                .ident = ma.ident,
            }},
            .subscript => |ss| return .{.subscript = .{
                .src = add_expr(try deep_copy_value(ss.src)),
                .idx = add_expr(try deep_copy_value(ss.idx)),
            }},
            .override => |o| return .{.override = .{
                .lhs = add_expr(try deep_copy_value(o.lhs)),
                .rhs = add_expr(try deep_copy_value(o.rhs)),
            }},
            .source_file => |sf| return .{.source_file = sf},
            .identifier => |i| return .{.identifier = i},
            .string => |s| return .{.string = s},
            .host => |h| return .{.host = h},
            .builtin_function => |f| return .{.builtin_function = f},
            .host_string_underlying => |hu| return .{.host_string_underlying = hu},

            .alias => |_| unreachable, // Expressions should not be resolved to aliases yet
        }
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
            .dict => |*s| s.parent = scope,
            .function => |*f| f.argument_template.parent = scope,
            .string, .alias, .host, .builtin_function, .host_string_underlying, .transformed_list, .joined_string,
            => {},
            .identifier => |i| {
                std.debug.assert(scope != NO_SCOPE);
                const is = tokens.at(i).identifier_string() catch unreachable;
                if(std.mem.eql(u8, is, "@host")) {
                    self.value = .{.host = self.bound()};
                    return;
                } else if(std.mem.eql(u8, is, "@transform")) {
                    self.value = .{.builtin_function = .{.token = i, .value = .transform}};
                    return;
                } else if(std.mem.eql(u8, is, "@join")) {
                    self.value = .{.builtin_function = .{.token = i, .value = .join}};
                    return;
                }
                return try self.make_alias(scope, expressions.at(scope).dealias().value.dict.lookup(is) orelse {
                    report_error(token_bound(i), "Identifier '{s}' not found!", .{is});
                    return error.BadIdentifier;
                });
            },
            .override => {
                @panic("TODO: Override");
            },
            .source_file => |sf| {
                if(source_files.at(sf.file).tokens == null) {
                    source_files.at(sf.file).tokens = try tokenize_file(sf.file);
                }
                if(source_files.at(sf.file).top_level_value == NO_SCOPE) {
                    var toks = source_files.at(sf.file).tokens.?;
                    source_files.at(sf.file).top_level_value = try parse_dict(&toks);
                }
                try self.make_alias(NO_SCOPE, source_files.at(sf.file).top_level_value);
            },
            .list => |l| {
                for(l.items) |expr| {
                    try expressions.at(expr).resolve(scope);
                }
            },
            .call => |c| {
                try expressions.at(c.callee).resolve(scope);
                switch(expressions.at(c.callee).dealias().value) {
                    .function => |f| {
                        var args = f.argument_template;
                        args.values = try args.values.clone(alloc);

                        {
                            var it = c.args.iterator();
                            while(it.next()) |a| {
                                try expressions.at(a.value_ptr.*).resolve(scope);
                                if((try args.values.fetchPut(alloc, a.key_ptr.*, a.value_ptr.*) orelse {
                                    report_error(
                                        expressions.at(a.value_ptr.*).bound(),
                                        "Unknown argument {s}",
                                        .{a.key_ptr.*},
                                    );
                                    return error.BadKey;
                                }).value != NO_ARG_VALUE) {
                                    report_error(
                                        expressions.at(a.value_ptr.*).bound(),
                                        "Argument {s} supplied twice",
                                        .{a.key_ptr.*},
                                    );
                                    return error.BadKey;
                                }
                            }
                        }
                        {
                            var it = args.values.iterator();
                            while(it.next()) |a| {
                                if(a.value_ptr.* == NO_ARG_VALUE) {
                                    report_error(
                                        self.bound(),
                                        "Missing argument {s}",
                                        .{a.key_ptr.*},
                                    );
                                    return error.MissingArgument;
                                }
                            }
                        }

                        const new_scope = add_expr(.{.dict = args});
                        self.value = try deep_copy_value(f.body);
                        self.resolving = false;
                        return @call(.always_tail, resolve, .{self, new_scope});
                    },
                    .builtin_function => |f| switch(f.value) {
                        .transform => {
                            std.debug.assert(c.args.size == 2);
                            const elements = c.args.get("elements").?;
                            const transform = c.args.get("transform").?;
                            try expressions.at(elements).resolve(scope);
                            try expressions.at(transform).resolve(scope);
                            self.value = .{.transformed_list = .{
                                .orig_bound = self.bound(),
                                .elements = elements,
                                .transform = transform,
                            }};
                        },
                        .join => {
                            std.debug.assert(c.args.size == 2);
                            const separator = c.args.get("separator").?;
                            const elements = c.args.get("elements").?;
                            try expressions.at(elements).resolve(scope);
                            try expressions.at(separator).resolve(scope);
                            self.value = .{.joined_string = .{
                                .orig_bound = self.bound(),
                                .elements = elements,
                                .separator = separator,
                                .scope = scope,
                            }};
                        },
                    },
                    else => {
                        report_error(
                            self.bound(),
                            "Expected function",
                            .{},
                        );
                        return error.BadValue;
                    }
                }
            },
            .member_access => |ma| {
                try expressions.at(ma.src).resolve(scope);
                const str = tokens.at(ma.ident).identifier_string() catch unreachable;
                try self.make_alias(ma.src, try expressions.at(ma.src).dealias().lookup(str) orelse {
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
                            try expressions.at(e).resolve(scope);
                        },
                        else => {},
                    }
                }
            },
            .subscript => |ss| {
                try expressions.at(ss.idx).resolve(scope);
                var str = std.ArrayListUnmanaged(u8){};
                defer str.deinit(alloc);
                try expressions.at(ss.idx).dealias().append_string_value(&str);

                try expressions.at(ss.src).resolve(scope);
                try self.make_alias(ss.src, try expressions.at(ss.src).dealias().lookup(str.items) orelse {
                    report_error(
                        expressions.at(ss.idx).bound(),
                        "Key '{s}' not found in dict",
                        .{str.items},
                    );
                    return error.BadKey;
                });
            },
        }
    }

    fn value_at_index(self: @This(), index: usize) !u32 {
        switch(self.dealias().value) {
            .list => |l| return l.items[index],
            .transformed_list => |t| {
                const element = try expressions.at(t.elements).dealias().value_at_index(index);
                const transform = try deep_copy_value(t.transform);
                const args = try transform.function.argument_template.values.clone(alloc);
                var value_it = args.valueIterator();
                value_it.next().?.* = element;
                std.debug.assert(args.size == 1);
                const new_scope = add_expr(.{.dict = .{.values = args, .parent = transform.function.argument_template.parent}});
                try expressions.at(transform.function.body).resolve(new_scope);
                return transform.function.body;
            },
            else => unreachable,
        }
    }

    fn append_string_value_at_index(self: @This(), index: usize, str: *std.ArrayListUnmanaged(u8)) !void {
        switch(self.dealias().value) {
            .list => |l| expressions.at(l.items[index]).dealias().append_string_value(str),
            .transformed_list => |t| expressions.at(try expressions.at(t.elements).value_at_index(index)).dealias().append_string_value(str),
            else => unreachable,
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
                            tokens.at(sc).append_string_value(str);
                        },
                        .expression_chunk => |ec| {
                            expressions.at(ec).append_string_value(str) catch unreachable;
                        },
                    }
                }
            },
            .host_string_underlying => {
                try str.appendSlice(alloc, host_path.?);
            },
            .joined_string => |j| {
                const separator = expressions.at(j.separator).dealias();
                try expressions.at(j.elements).resolve(j.scope);
                switch(expressions.at(j.elements).dealias().value) {
                    .list => |l| for(l.items, 0..) |item, i| {
                        if(i > 0) try separator.append_string_value(str);
                        try expressions.at(item).dealias().append_string_value(str);
                    },
                    .transformed_list => |t| {
                        var current = t.elements;
                        while(expressions.at(current).dealias().value == .transformed_list) {
                            current = expressions.at(current).dealias().value.transformed_list.elements;
                        }
                        const transformed_list = expressions.at(j.elements).dealias();
                        const list = expressions.at(current).dealias();
                        std.debug.assert(list.value == .list);
                        for(0..list.value.list.items.len) |i| {
                            if(i > 0) try separator.append_string_value(str);
                            try expressions.at(try transformed_list.value_at_index(i)).dealias().append_string_value(str);
                        }
                    },
                    else => unreachable,
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
            .host_string_underlying => unreachable,

            .string_parts => |sp| {
                var result = @as(?SourceBound, null);
                for(sp) |part| {
                    var part_bound = @as(SourceBound, undefined);
                    switch(part) {
                        .string_chunk => |sc| part_bound = token_bound(sc),
                        .expression_chunk => |ec| part_bound = expressions.at(ec).bound(),
                    }
                    if(result) |res| {
                        result = res.combine(part_bound);
                    } else {
                        result = part_bound;
                    }
                }
                return result.?;
            },

            .transformed_list => |t| t.orig_bound,
            .joined_string => |j| j.orig_bound,

            .member_access => |ma| return expressions.at(ma.src).bound().combine(token_bound(ma.ident)),

            .list => |l| {
                var result = l.lsquare_bound;
                for(l.items) |item| {
                    result = result.combine(expressions.at(item).bound());
                }
                return result.extend_right(1);
            },

            .call => |fc| {
                var result = expressions.at(fc.callee).bound();
                var iter = fc.args.valueIterator();
                while(iter.next()) |arg| {
                    result = result.combine(expressions.at(arg.*).bound());
                }
                if(result.peek_token()) |token| {
                    if(token.value == .comma) result = result.extend_right(1);
                }
                return result.extend_right(1);
            },

            .function => |func| return expressions.at(func.body).bound(),

            .dict => |dict| {
                var result = @as(?SourceBound, null);
                var iter = dict.values.valueIterator();
                while(iter.next()) |value| {
                    if(result) |res| {
                        result = res.combine(expressions.at(value.*).bound());
                    } else {
                        result = expressions.at(value.*).bound();
                    }
                }
                return result.?;
            },

            .override => |o| expressions.at(o.lhs).bound().combine(expressions.at(o.rhs).bound()),

            .subscript => |ss| expressions.at(ss.src).bound().combine(
                expressions.at(ss.idx).bound().extend_right(1),
            ),

            .source_file => |sf| sf.bound,

            inline .string, .alias => |v| v.orig_bound,
            .host => |b| b,
            .builtin_function => |f| token_bound(f.token),
            .identifier,
            => |t| token_bound(t),
        };
    }
};

pub fn main() !void {
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
    try expressions.at(root_expr).resolve(NO_SCOPE);

    // Ugly hack to parse cmdline as expression
    try source_files.append(alloc, .{
        .buffer = positional_args.slice()[2],
        .realpath = "<cmdline>",
        .dir = std.fs.cwd(),
        .top_level_value = NO_SCOPE,
        .tokens = null,
    });
    var cmdline_tokens = try tokenize_file(@intCast(u32, source_files.count() - 1));
    const cmdline_expr = try parse_expr(&cmdline_tokens);
    try expressions.at(cmdline_expr).resolve(root_expr);
    const result = try expressions.at(cmdline_expr).to_string();
    std.debug.print("{s}\n",.{result});
}
